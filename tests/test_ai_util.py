"""AI 标注单元测试：并发回调口径、进度单调性、失败隔离与日志回调"""

import threading
import time

import pytest

from src import ai_util


def _patch(monkeypatch, fail_ids=(), raise_ids=()):
    def fake_chat(base_url, api_key, model, messages, **kw):
        img = messages[1]["content"][1]["image_url"]["url"]
        mid = img.rsplit("/", 1)[-1]
        if mid in raise_ids:
            raise RuntimeError("boom " + mid)
        if mid in fail_ids:
            return "这不是 JSON"
        return '{"name": "名-' + mid + '", "ocr": "文-' + mid + '"}'

    monkeypatch.setattr(ai_util, "chat_completion", fake_chat)
    monkeypatch.setattr(
        ai_util,
        "encode_image_base64",
        lambda path, max_px=512: "data:image/png;base64,/" + path,
    )


def _memes(n):
    return [{"id": i, "path": "id-" + str(i)} for i in range(1, n + 1)]


def test_ai_tag_memes_counts_every_item(monkeypatch):
    """进度回调口径与总数一致：成功与失败都计入 done；结果集只含成功条目"""
    _patch(monkeypatch, fail_ids={"id-2"})
    seen = []

    results = ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(5), on_progress=lambda d, t: seen.append((d, t))
    )

    assert [d for d, _ in seen] == [1, 2, 3, 4, 5]
    assert {t for _, t in seen} == {5}
    # 失败条目不计入结果，避免前端出现空名称的待确认行
    assert [r["id"] for r in results] == [1, 3, 4, 5]
    assert all(not r["error"] for r in results)
    assert results[0]["name"] == "名-id-1"
    assert results[0]["ocr"] == "文-id-1"


def test_ai_tag_memes_progress_is_monotonic(monkeypatch):
    """并发场景下 done 序列必须严格递增，不得回退（否则前端进度条来回跳）"""
    _patch(monkeypatch)
    done_seq = []
    lock = threading.Lock()

    def on_progress(done, total):
        with lock:
            done_seq.append(done)

    ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(32), on_progress=on_progress, concurrency=8
    )

    assert done_seq == sorted(done_seq)
    assert done_seq[-1] == 32


def test_ai_tag_memes_on_result_reports_failures(monkeypatch):
    """每条结果都回调 on_result，失败条目带 error 文案"""
    _patch(monkeypatch, fail_ids={"id-3"}, raise_ids={"id-4"})
    got = []

    ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(4), on_result=lambda r: got.append(r)
    )

    by_id = {r["id"]: r for r in got}
    assert set(by_id) == {1, 2, 3, 4}
    assert by_id[1]["error"] == "" and by_id[1]["name"] == "名-id-1"
    assert by_id[3]["error"] == "AI 未返回有效名称"
    assert "boom id-4" in by_id[4]["error"]


def test_ai_tag_memes_empty_input(monkeypatch):
    """空列表直接返回，不触发任何回调"""
    calls = []
    results = ai_util.ai_tag_memes(
        "http://x", "k", "m", [], on_progress=lambda *a: calls.append(a)
    )
    assert results == [] and calls == []


def test_ai_tag_memes_stop_breaks_early(monkeypatch):
    """should_stop 生效时剩余条目不进入结果"""
    _patch(monkeypatch)
    stop = [False]

    def on_result(r):
        stop[0] = True

    results = ai_util.ai_tag_memes(
        "http://x",
        "k",
        "m",
        _memes(20),
        on_result=on_result,
        should_stop=lambda: stop[0],
        concurrency=1,
    )

    assert len(results) < 20


def test_clean_display_name_trims_and_strips():
    # 换行/制表符属非法文件名字符，先去空
    assert ai_util.clean_display_name("  a\n\nb  ") == "ab"
    assert ai_util.clean_display_name("a   b") == "a b"
    assert ai_util.clean_display_name("  a/b:c  ") == "abc"
    assert ai_util.clean_display_name("", "兜底") == "兜底"
    assert ai_util.clean_display_name(None, "兜底") == "兜底"
    assert len(ai_util.clean_display_name("字" * 100)) == ai_util._NAME_MAX


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"name":"n","ocr":"o"}', {"name": "n", "ocr": "o"}),
        ('```json\n{"name":"n"}\n```', {"name": "n"}),
        ('好的，结果如下 {"name":"n"} 完毕', {"name": "n"}),
        ("不是 JSON", None),
        ("", None),
    ],
)
def test_parse_ai_json_levels(text, expected):
    assert ai_util._parse_ai_json(text) == expected


# ─── 错误分类 ───


@pytest.mark.parametrize(
    "status,kind",
    [
        (401, "fatal"),
        (402, "fatal"),
        (403, "fatal"),
        (404, "fatal"),
        (429, "retryable"),
        (500, "retryable"),
        (502, "retryable"),
        (503, "retryable"),
        (504, "retryable"),
        (408, "retryable"),
        (599, "retryable"),
        (400, "transient"),
        (422, "transient"),
    ],
)
def test_classify_status(status, kind):
    assert ai_util._classify_status(status) == kind


def test_ai_service_error_flags():
    fatal = ai_util.AIServiceError("bad key", kind="fatal", status=401)
    assert fatal.fatal and not fatal.retryable
    retryable = ai_util.AIServiceError("limited", kind="retryable", status=429)
    assert retryable.retryable and not retryable.fatal
    transient = ai_util.AIServiceError("jitter", kind="transient")
    assert transient.retryable and not transient.fatal
    local = ai_util.AIServiceError("no file", kind="local")
    assert not local.retryable and not local.fatal


def _http_error(status, retry_after=None):
    """构造带状态码的 HTTPError，用于测试 _open_json 的分类"""
    import email.message
    import io as _io
    import urllib.error

    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "http://x/v1/chat/completions",
        status,
        "err",
        headers,
        _io.BytesIO(b'{"error":"nope"}'),
    )


def test_open_json_preserves_status(monkeypatch):
    """HTTP 状态码必须保留到异常上，否则上层无法区分 401 与 429"""
    import urllib.request

    def boom(req, timeout=None):
        raise _http_error(429, retry_after=7)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ai_util.AIServiceError) as ei:
        ai_util._open_json("http://x/v1/models", "k")
    err = ei.value
    assert err.status == 429
    assert err.kind == "retryable"
    assert err.retry_after == 7


def test_open_json_fatal_on_401(monkeypatch):
    import urllib.request

    def boom(req, timeout=None):
        raise _http_error(401)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ai_util.AIServiceError) as ei:
        ai_util._open_json("http://x/v1/models", "k")
    assert ei.value.fatal and ei.value.status == 401


# ─── 退避重试 ───


def test_chat_completion_retries_on_retryable(monkeypatch):
    """可重试错误应退避后重试，成功即返回，不把异常抛给上层"""
    monkeypatch.setattr(ai_util, "_sleep_backoff", lambda *a, **k: True)
    calls = {"n": 0}

    def fake_open(url, key, payload=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ai_util.AIServiceError("limited", kind="retryable", status=429)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(ai_util, "_open_json", fake_open)
    out = ai_util.chat_completion("http://x", "k", "m", [])
    assert out == "ok" and calls["n"] == 3


def test_chat_completion_does_not_retry_fatal(monkeypatch):
    """致命错误必须立即抛出，一次都不能重试"""
    monkeypatch.setattr(ai_util, "_sleep_backoff", lambda *a, **k: True)
    calls = {"n": 0}

    def fake_open(url, key, payload=None, timeout=None):
        calls["n"] += 1
        raise ai_util.AIServiceError("bad key", kind="fatal", status=401)

    monkeypatch.setattr(ai_util, "_open_json", fake_open)
    with pytest.raises(ai_util.AIServiceError) as ei:
        ai_util.chat_completion("http://x", "k", "m", [])
    assert calls["n"] == 1 and ei.value.fatal


def test_chat_completion_gives_up_after_max_retries(monkeypatch):
    """重试次数用尽后仍失败，则抛出最后一次错误"""
    monkeypatch.setattr(ai_util, "_sleep_backoff", lambda *a, **k: True)
    calls = {"n": 0}

    def fake_open(url, key, payload=None, timeout=None):
        calls["n"] += 1
        raise ai_util.AIServiceError("down", kind="retryable", status=503)

    monkeypatch.setattr(ai_util, "_open_json", fake_open)
    with pytest.raises(ai_util.AIServiceError):
        ai_util.chat_completion("http://x", "k", "m", [], retries=2)
    assert calls["n"] == 3  # 首次 + 2 次重试


def test_chat_completion_retries_bad_structure(monkeypatch):
    """响应结构异常属内容抖动，应重试而非直接失败"""
    monkeypatch.setattr(ai_util, "_sleep_backoff", lambda *a, **k: True)
    calls = {"n": 0}

    def fake_open(url, key, payload=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"unexpected": True}
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(ai_util, "_open_json", fake_open)
    assert ai_util.chat_completion("http://x", "k", "m", []) == "ok"
    assert calls["n"] == 2


def test_sleep_backoff_respects_stop():
    """退避期间收到中止信号应立即返回 False，不睡满"""
    stop = {"v": False}
    threading.Timer(0.05, lambda: stop.update(v=True)).start()
    t0 = time.time()
    ok = ai_util._sleep_backoff(3, retry_after=5, should_stop=lambda: stop["v"])
    assert ok is False and (time.time() - t0) < 2


# ─── 熔断 ───


def test_breaker_trips_on_fatal_immediately():
    brk = ai_util.CircuitBreaker()
    err = ai_util.AIServiceError("bad key", kind="fatal", status=401)
    assert brk.record(False, err) is True
    assert brk.tripped and "重试无意义" in brk.reason
    # 已熔断后再登记不重复触发
    assert brk.record(False, err) is False


def test_breaker_trips_on_consecutive_failures():
    brk = ai_util.CircuitBreaker(consecutive_limit=3)
    err = ai_util.AIServiceError("down", kind="retryable", status=503)
    assert brk.record(False, err) is False
    assert brk.record(False, err) is False
    assert brk.record(False, err) is True
    assert "连续 3 次失败" in brk.reason


def test_breaker_success_resets_consecutive():
    brk = ai_util.CircuitBreaker(consecutive_limit=3)
    err = ai_util.AIServiceError("down", kind="retryable", status=503)
    brk.record(False, err)
    brk.record(False, err)
    brk.record(True)
    brk.record(False, err)
    brk.record(False, err)
    assert brk.tripped is False


def test_breaker_trips_on_low_success_rate():
    brk = ai_util.CircuitBreaker(consecutive_limit=99, min_tried=10, min_success=0.5)
    err = ai_util.AIServiceError("down", kind="retryable", status=503)
    for _ in range(9):
        brk.record(False, err)
    assert brk.tripped is False  # 未达 min_tried
    brk.record(True)
    brk.record(False, err)
    assert brk.tripped is True
    assert "成功率" in brk.reason


def test_breaker_stats_shape():
    brk = ai_util.CircuitBreaker()
    brk.record(True)
    brk.record(False, ai_util.AIServiceError("x", kind="retryable"))
    s = brk.stats
    assert s["tried"] == 2 and s["ok"] == 1 and s["consecutive_failed"] == 1
    assert s["tripped"] is False


def test_ai_tag_memes_trips_breaker_and_stops_early(monkeypatch):
    """致命错误下应触发熔断、abort 队列，且 on_breaker 只回调一次"""
    calls = {"n": 0}

    def fake_chat(base_url, api_key, model, messages, **kw):
        calls["n"] += 1
        raise ai_util.AIServiceError("bad key", kind="fatal", status=401)

    monkeypatch.setattr(ai_util, "chat_completion", fake_chat)
    monkeypatch.setattr(
        ai_util, "encode_image_base64", lambda p, max_px=512: "data:image/x;base64,AA"
    )
    reasons = []
    results = ai_util.ai_tag_memes(
        "http://x",
        "k",
        "m",
        _memes(50),
        on_breaker=reasons.append,
        concurrency=1,
    )
    assert results == []
    assert len(reasons) == 1
    # 熔断后不再继续打完 50 张
    assert calls["n"] < 50


def test_ai_tag_memes_local_error_does_not_trip(monkeypatch):
    """读不到本地图片属本地问题，不该把整个任务熔断掉"""
    monkeypatch.setattr(ai_util, "encode_image_base64", lambda p, max_px=512: "")
    reasons = []
    results = ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(30), on_breaker=reasons.append, concurrency=1
    )
    assert results == [] and reasons == []


def test_ai_tag_memes_content_error_does_not_trip(monkeypatch):
    """AI 返回内容不可解析≠服务故障，不该熔断"""
    calls = {"n": 0}

    def fake_chat(base_url, api_key, model, messages, **kw):
        calls["n"] += 1
        return "这不是 JSON"

    monkeypatch.setattr(ai_util, "chat_completion", fake_chat)
    monkeypatch.setattr(
        ai_util, "encode_image_base64", lambda p, max_px=512: "data:image/x;base64,AA"
    )
    reasons = []
    ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(30), on_breaker=reasons.append, concurrency=1
    )
    assert reasons == [] and calls["n"] == 30


def test_ai_tag_memes_reports_kind_in_result(monkeypatch):
    """失败结果要带 kind，供上层区分「服务挂了」和「这张图内容不行」"""
    seen = []

    def fake_chat(base_url, api_key, model, messages, **kw):
        raise ai_util.AIServiceError("down", kind="retryable", status=503)

    monkeypatch.setattr(ai_util, "chat_completion", fake_chat)
    monkeypatch.setattr(
        ai_util, "encode_image_base64", lambda p, max_px=512: "data:image/x;base64,AA"
    )
    ai_util.ai_tag_memes(
        "http://x", "k", "m", _memes(3), on_result=seen.append, concurrency=1
    )
    assert seen and seen[0].get("kind") == "retryable"
