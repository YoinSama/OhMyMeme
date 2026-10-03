"""AI 标注运行时验证：任务互斥、熔断中止、同 hash 结果扇出

与 test_ai_util.py 不同，这里直接跑 src.webui 的 worker，覆盖
「启动入口 → 收集目标 → 调 AI → 落建议 → 收尾状态」的完整链路。
"""

import json
import threading
import time

import pytest

from src import ai_util, webui


class FakeResponse:
    """最小 HTTP 响应替身"""

    def __init__(self, payload=None, status=200, retry_after=None):
        self.status = status
        self._body = json.dumps(payload if payload is not None else {}).encode()
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _meme(mid, filename, fhash=""):
    return {
        "id": mid,
        "filename": filename,
        "file_hash": fhash,
    }


@pytest.fixture
def fake_env(monkeypatch):
    """把 db / 文件查找 / 配置 / AI 调用全部换成可控替身"""
    state = {
        "rows": [],
        "config": {
            "ai_base_url": "http://x",
            "ai_api_key": "k",
            "ai_model": "m",
            "ai_concurrency": 4,
        },
    }

    class FakeDb:
        def get_by_id(self, mid):
            for r in state["rows"]:
                if r["id"] == mid:
                    return dict(r)
            return None

        def search(self, **kw):
            return [dict(r) for r in state["rows"]]

    monkeypatch.setattr(webui, "get_db", lambda: FakeDb())
    monkeypatch.setattr(webui, "get_config", lambda: dict(state["config"]))

    class FakeWebui:
        """_find_meme_file 是实例方法，worker 通过传入的 app 对象调用它"""

        def _find_meme_file(self, fname):
            return "C:/fake/" + fname

    state["app"] = FakeWebui()
    # 隔离真实库表：建议写入 _AI_SUGGESTIONS 是本模块内存字典，直接清空即可
    monkeypatch.setattr(webui, "_AI_SUGGESTIONS", {})
    yield state
    webui._AI_SUGGESTIONS.clear()
    with webui._AI_LOCK:
        webui._AI_TASK["task_id"] = ""
        webui._AI_TASK["thread"] = None
        webui._AI_TASK["cancel"] = None
    webui._set_ai(status="idle", progress=0, ok=0, failed=0, error="")


def _wait_done(timeout=10.0):
    """等当前任务跑完"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not webui.ai_task_running():
            return True
        time.sleep(0.02)
    return False


def test_task_mutex_second_start_rejected(fake_env, monkeypatch):
    """第一个任务运行期间，第二次启动被拒绝并返回同一个 task_id"""
    fake_env["rows"] = [_meme(i, "f%d.png" % i, "h%d" % i) for i in range(1, 6)]
    started = threading.Event()
    release = threading.Event()

    def slow_tag(*a, **kw):
        started.set()
        release.wait(5)
        return [{"id": t["id"], "name": "n", "ocr": ""} for t in a[3]]

    monkeypatch.setattr(ai_util, "ai_tag_memes", slow_tag)

    first = webui.start_ai_tag(fake_env["app"])
    assert started.wait(5), "第一个任务未能启动"
    assert webui.ai_task_running() is True

    second = webui.start_ai_tag(fake_env["app"])
    assert second == first, "第二个任务没有被拒绝，返回了不同的 task_id"
    assert webui.running_ai_task_id() == first

    release.set()
    assert _wait_done(), "任务未在超时内结束"
    # 任务结束后槽位释放，可以重新启动
    assert webui.ai_task_running() is False
    assert webui.running_ai_task_id() == ""


def test_breaker_aborts_task_and_keeps_success(fake_env, monkeypatch):
    """致命错误触发熔断：任务状态为 error，已成功建议仍保留"""
    fake_env["rows"] = [_meme(i, "f%d.png" % i, "h%d" % i) for i in range(1, 4)]

    def boom(base_url, api_key, model, targets, **kw):
        breaker = kw["breaker"]
        on_breaker = kw["on_breaker"]
        breaker.record(True)
        breaker.record(False, ai_util.AIServiceError("401 未授权", kind="fatal"))
        on_breaker("API 鉴权失败")
        return [{"id": targets[0]["id"], "name": "成功的那张", "ocr": ""}]

    monkeypatch.setattr(ai_util, "ai_tag_memes", boom)

    task_id = webui.start_ai_tag(fake_env["app"])
    assert task_id
    assert _wait_done(), "任务未在超时内结束"

    st = webui.get_ai_progress()
    assert st["status"] == "error", "熔断后状态应为 error，实际 %s" % st["status"]
    assert "API 鉴权失败" in st["error"]
    store = webui._AI_SUGGESTIONS.get(task_id, {})
    assert len(store) >= 1, "熔断中止时已成功的建议丢失了"


def test_dup_hash_fanout(fake_env, monkeypatch):
    """同 file_hash 的多条记录只发一次请求，结果扇出到全部同图 id"""
    fake_env["rows"] = [
        _meme(1, "a.png", "SAME"),
        _meme(2, "a_copy.png", "SAME"),
        _meme(3, "a_copy2.png", "SAME"),
        _meme(4, "b.png", "OTHER"),
    ]
    calls = []

    def spy(base_url, api_key, model, targets, **kw):
        calls.append([t["id"] for t in targets])
        return [{"id": t["id"], "name": "名%d" % t["id"], "ocr": ""} for t in targets]

    monkeypatch.setattr(ai_util, "ai_tag_memes", spy)

    task_id = webui.start_ai_tag(fake_env["app"])
    assert _wait_done(), "任务未在超时内结束"

    # 4 条记录、2 个不同 hash → 只应发出 2 次请求（id 1 与 id 4）
    assert calls == [[1, 4]], "去重后请求目标不正确：%s" % calls
    store = webui._AI_SUGGESTIONS.get(task_id, {})
    assert set(store) == {"1", "2", "3", "4"}, "扇出后建议应覆盖全部同图记录"
    assert store["2"]["name"] == "名1", "重复记录未拿到与主记录一致的名字"
    assert store["3"]["name"] == "名1"
    assert store["4"]["name"] == "名4"


def test_cancel_marks_task_cancelled(fake_env, monkeypatch):
    """取消请求能把任务收敛到 cancelled 状态并保留已有建议"""
    fake_env["rows"] = [_meme(i, "f%d.png" % i, "h%d" % i) for i in range(1, 9)]
    hold = threading.Event()

    def cancelling(base_url, api_key, model, targets, **kw):
        kw["on_result"]({"id": targets[0]["id"], "name": "先成一张", "ocr": ""})
        hold.wait(5)
        should_stop = kw["should_stop"]
        while not should_stop():
            time.sleep(0.02)
        return [{"id": targets[0]["id"], "name": "先成一张", "ocr": ""}]

    monkeypatch.setattr(ai_util, "ai_tag_memes", cancelling)

    task_id = webui.start_ai_tag(fake_env["app"])
    assert task_id
    time.sleep(0.15)
    assert webui.cancel_ai_task(task_id) is True
    hold.set()
    assert _wait_done(), "取消后任务未收敛"

    st = webui.get_ai_progress()
    assert st["status"] == "cancelled", "实际状态 %s" % st["status"]
    assert webui._AI_SUGGESTIONS.get(task_id), "取消后应保留已完成的建议"


def test_cancel_with_wrong_task_id_ignored(fake_env, monkeypatch):
    """task_id 不匹配的取消请求不生效，避免误伤当前任务"""
    fake_env["rows"] = [_meme(i, "f%d.png" % i, "h%d" % i) for i in range(1, 5)]
    hold = threading.Event()

    def slow(base_url, api_key, model, targets, **kw):
        hold.wait(5)
        return [{"id": t["id"], "name": "n", "ocr": ""} for t in targets]

    monkeypatch.setattr(ai_util, "ai_tag_memes", slow)

    task_id = webui.start_ai_tag(fake_env["app"])
    assert task_id
    time.sleep(0.1)
    assert webui.cancel_ai_task("not-the-current-task") is False
    assert webui.get_ai_progress()["status"] == "running"

    assert webui.cancel_ai_task(task_id) is True
    hold.set()
    assert _wait_done()
