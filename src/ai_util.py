# -*- coding: utf-8 -*-
"""AI 标注工具模块

通过 OpenAI 兼容接口为表情包自动生成显示名（「标签-内容描述」）与图上文字。
兼容任意 OpenAI 格式的服务商与中转站，全部使用标准库，不引入外部依赖。

对外函数：
- list_models: 拉取可用模型列表（GET /v1/models），视觉模型置顶
- chat_completion: 多模态对话
- encode_image_base64: 图片缩放后转 base64 data URI
- ai_tag_memes: 并发批量标注，返回每条的名称与 OCR 文字
"""

import base64
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# HTTP 请求超时（秒）
_HTTP_TIMEOUT = 60
# 模型列表请求超时（秒）
_MODELS_TIMEOUT = 20
# 单次标注最大 token
_MAX_TOKENS = 300
# 送检图片最长边像素（超过则缩放，省 token）
_MAX_IMAGE_PX = 512
# 显示名最大长度
_NAME_MAX = 60

# 重试：首次失败后最多再试的次数
_MAX_RETRIES = 2
# 重试初始退避（秒）
_RETRY_BACKOFF = 1.0
# 退避抖动比例（±30%，避免并发请求同步重试再一起撞限流）
_RETRY_JITTER = 0.3

# 熔断阈值：连续失败多少次即中止整个任务
_BREAKER_CONSECUTIVE = 5
# 熔断阈值：已试满多少张后才启用成功率判定
_BREAKER_MIN_TRIED = 20
# 熔断阈值：成功率低于此值即中止（0-1）
_BREAKER_MIN_SUCCESS = 0.5

# 致命状态码：密钥/余额/地址问题，任何一张失败就立即熔断，重试毫无意义
_FATAL_STATUS = (401, 402, 403, 404)
# 可重试状态码：限流与服务端临时故障
_RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)

# Windows 与通用文件系统均不允许的字符
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')
_WHITESPACE = re.compile(r"\s+")

_STYLE_NAMES = {
    "general": "通用聊天",
    "anime": "二次元",
    "work": "职场",
    "gaming": "游戏群",
}

# 视觉（多模态）模型名的常见特征，用于列表排序置顶
_VISION_HINTS = (
    "vl",
    "vision",
    "4o",
    "4v",
    "gemini",
    "claude-3",
    "qwen-vl",
    "glm-4v",
    "internvl",
    "minicpm-v",
)

_SYSTEM_PROMPT = (
    "你是表情包标注助手。分析图片后只返回一个 JSON："
    '{"name":"标签-内容描述","ocr":"图上文字"}'
    "\nname 规则：标签为 1-2 个汉字，概括情绪或动作"
    "（如 笑、哭、无语、点赞、摊手），紧跟一个半角减号 - ，"
    "再跟 10-20 字客观描述画面内容。示例：笑-橘猫在沙发上打滚"
    "\nocr 规则：图上可见文字原样抄录，没有文字则填空字符串"
    '\n禁止出现 / \\ : * ? " < > | 与换行。只返回 JSON，不要其他文字。'
    "\n标注风格为%s，按该场景选择用词。"
)


def _normalize_base_url(base_url):
    """规范化 API 地址：缺 /v1 时补齐，已带 /v1 时不重复拼接

    中转站常见两种写法（https://x.com 与 https://x.com/v1），
    直接硬拼 /v1 会产生 /v1/v1/chat/completions 导致 404，故在此统一。
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return ""
    if not url.endswith("/v1"):
        url += "/v1"
    return url


class AIServiceError(ValueError):
    """AI 服务调用失败，带分类信息供上层决定重试还是熔断

    kind 取值：
    - fatal:    密钥/余额/地址问题，重试无意义，应立刻中止整个任务
    - retryable: 限流或服务端临时故障，退避后可重试
    - transient: 连接层异常或响应不可解析，可尝试重试一次
    - local:     本地问题（文件读不了等），与网络无关，不该计入熔断
    """

    def __init__(self, message, kind="transient", status=0, retry_after=0):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after = retry_after

    @property
    def fatal(self):
        return self.kind == "fatal"

    @property
    def retryable(self):
        return self.kind in ("retryable", "transient")


def _classify_status(status, retry_after=0):
    """按 HTTP 状态码归类错误"""
    if status in _FATAL_STATUS:
        return "fatal"
    if status in _RETRYABLE_STATUS:
        return "retryable"
    if status >= 500:
        return "retryable"
    return "transient"


def _parse_retry_after(resp):
    """读取 429/503 响应头里的 Retry-After，返回秒数（取不到为 0）"""
    try:
        raw = resp.headers.get("Retry-After")
    except Exception:
        return 0
    if not raw:
        return 0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0


def _open_json(url, api_key, payload=None, timeout=_HTTP_TIMEOUT):
    """发送 JSON 请求并返回解析结果，payload 为空时用 GET

    失败一律抛 AIServiceError，并保留 HTTP 状态码与错误分类，
    供 ai_tag_memes 决定重试或熔断（早期版本把所有异常压成一句
    "请求失败"，上层无法区分 401 与 429）。
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data is not None else "GET"
    )
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer %s" % api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = int(getattr(e, "code", 0) or 0)
        retry_after = _parse_retry_after(e)
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        hint = "%s %s" % (status, detail) if detail else str(status)
        raise AIServiceError(
            "HTTP %s: %s" % (hint, e.reason if e.reason else ""),
            kind=_classify_status(status, retry_after),
            status=status,
            retry_after=retry_after,
        )
    except urllib.error.URLError as e:
        raise AIServiceError("连接失败: %s" % e.reason, kind="transient")
    except TimeoutError:
        raise AIServiceError("请求超时（%ss）" % timeout, kind="retryable")
    except Exception as e:
        raise AIServiceError("请求失败: %s" % e, kind="transient")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise AIServiceError("响应不是有效 JSON: %s" % body[:200], kind="transient")


def list_models(base_url, api_key, timeout=_MODELS_TIMEOUT):
    """拉取可用模型列表，视觉模型排在前面

    返回 [{"id": 模型名, "vision": 是否疑似多模态}, ...]
    """
    url = _normalize_base_url(base_url)
    if not url:
        raise ValueError("请填写 API 地址")
    data = _open_json(url + "/models", api_key, timeout=timeout)
    raw = data.get("data") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("模型列表响应异常: %s" % json.dumps(data)[:200])
    seen = set()
    models = []
    for item in raw:
        mid = ""
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or ""
        elif isinstance(item, str):
            mid = item
        mid = str(mid).strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        low = mid.lower()
        models.append({"id": mid, "vision": any(hint in low for hint in _VISION_HINTS)})
    models.sort(key=lambda m: (not m["vision"], m["id"]))
    return models


def _sleep_backoff(attempt, retry_after=0, should_stop=None):
    """按指数退避 + 抖动睡眠，返回是否应继续（被中止则 False）"""
    if retry_after > 0:
        delay = min(retry_after, 30.0)
    else:
        base = _RETRY_BACKOFF * (2**attempt)
        jitter = base * _RETRY_JITTER
        delay = base + random.uniform(-jitter, jitter)
    delay = max(0.0, delay)
    # 分片睡眠，便于及时响应中止
    step = 0.2
    waited = 0.0
    while waited < delay:
        if should_stop and should_stop():
            return False
        chunk = min(step, delay - waited)
        time.sleep(chunk)
        waited += chunk
    return True


def chat_completion(
    base_url,
    api_key,
    model,
    messages,
    max_tokens=_MAX_TOKENS,
    retries=_MAX_RETRIES,
    should_stop=None,
):
    """调用 /v1/chat/completions，返回 content 文本

    可重试的错误（限流、5xx、连接抖动）按指数退避重试；
    致命错误（密钥/余额/地址）立即抛出，由上层熔断。
    """
    if not model:
        raise AIServiceError("模型不能为空", kind="fatal")
    url = _normalize_base_url(base_url)
    if not url:
        raise AIServiceError("请填写 API 地址", kind="fatal")
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    last_err = None
    for attempt in range(max(0, int(retries)) + 1):
        if should_stop and should_stop():
            raise AIServiceError("已取消", kind="local")
        try:
            data = _open_json(url + "/chat/completions", api_key, payload)
        except AIServiceError as e:
            last_err = e
            if e.fatal or not e.retryable or attempt >= retries:
                raise
            if not _sleep_backoff(attempt, e.retry_after, should_stop):
                raise AIServiceError("已取消", kind="local")
            continue
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # 结构异常多为内容抖动，重试一次通常能恢复
            last_err = AIServiceError(
                "AI 响应结构异常: %s" % json.dumps(data)[:200], kind="transient"
            )
            if attempt >= retries:
                raise last_err
            if not _sleep_backoff(attempt, 0, should_stop):
                raise AIServiceError("已取消", kind="local")
    raise last_err or AIServiceError("请求失败", kind="transient")


def encode_image_base64(path, max_px=_MAX_IMAGE_PX):
    """读取图片，按需缩放后转 base64 data URI

    动图取中间帧（比首帧更能代表内容），无 Pillow 时退化为原图直传。
    """
    try:
        from PIL import Image
    except ImportError:
        Image = None

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    if ext not in ("png", "jpeg", "gif", "webp", "bmp"):
        ext = "png"
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return ""

    if Image is not None:
        try:
            with Image.open(io.BytesIO(raw)) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames > 1:
                    img.seek(min(n_frames - 1, n_frames // 2))
                if max(img.size) > max_px:
                    img.thumbnail((max_px, max_px), Image.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, "JPEG", quality=80)
            compressed = buf.getvalue()
            # 仅当压缩后确实更小时才采用，避免小图反而变大
            if compressed and len(compressed) < len(raw):
                raw = compressed
                ext = "jpeg"
        except Exception:
            pass

    return "data:image/%s;base64,%s" % (ext, base64.b64encode(raw).decode("ascii"))


def _parse_ai_json(text):
    """宽松解析 AI 返回的 JSON（可能带 markdown 代码块或多余文字）"""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def clean_display_name(text, fallback=""):
    """清洗 AI 生成的文本：去非法字符、压空白、限长"""
    if not isinstance(text, str):
        return fallback
    s = _INVALID_CHARS.sub("", text).strip()
    s = _WHITESPACE.sub(" ", s)
    if len(s) > _NAME_MAX:
        s = s[:_NAME_MAX]
    return s or fallback


def _build_messages(data_uri, style_text):
    return [
        {"role": "system", "content": _SYSTEM_PROMPT % style_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "分析这张表情包，按要求返回 JSON。"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]


class CircuitBreaker:
    """任务级熔断器：任务内共享，不跨任务保留状态

    标注是用户手动触发的批处理而非持续流量，跨任务持久化熔断器会带来
    「进程重启后如何恢复」「改了密钥怎么解除」等额外状态管理，不值得。
    因此熔断状态的生命周期 = 一次 ai_tag_memes 调用。

    两种熔断：
    - 致命错误（密钥/余额/地址）：第一次就中止，重试毫无意义
    - 阈值错误：连续失败 N 次，或试满一定数量后成功率过低
    """

    def __init__(
        self,
        consecutive_limit=_BREAKER_CONSECUTIVE,
        min_tried=_BREAKER_MIN_TRIED,
        min_success=_BREAKER_MIN_SUCCESS,
    ):
        self._lock = threading.Lock()
        self._consecutive = 0
        self._tried = 0
        self._ok = 0
        self._tripped = False
        self._reason = ""
        self.consecutive_limit = max(1, int(consecutive_limit))
        self.min_tried = max(1, int(min_tried))
        self.min_success = float(min_success)

    @property
    def tripped(self):
        with self._lock:
            return self._tripped

    @property
    def reason(self):
        """熔断原因文案，未熔断为空串"""
        with self._lock:
            return self._reason

    @property
    def stats(self):
        with self._lock:
            return {
                "tried": self._tried,
                "ok": self._ok,
                "consecutive_failed": self._consecutive,
                "tripped": self._tripped,
                "reason": self._reason,
            }

    def record(self, ok, err=None):
        """登记一次结果，返回是否刚刚触发熔断

        ok 为 True 时重置连续失败计数；err 为 AIServiceError 且致命则立即熔断。
        """
        with self._lock:
            if self._tripped:
                return False
            self._tried += 1
            if ok:
                self._ok += 1
                self._consecutive = 0
                return False
            self._consecutive += 1
            fatal = bool(getattr(err, "fatal", False))
            if fatal:
                self._tripped = True
                self._reason = "服务端拒绝了请求，重试无意义：%s" % (
                    err or "凭证或地址有误"
                )
                return True
            if self._consecutive >= self.consecutive_limit:
                self._tripped = True
                self._reason = "连续 %d 次失败，疑似服务不可用" % self._consecutive
                return True
            if self._tried >= self.min_tried:
                rate = self._ok / float(self._tried)
                if rate < self.min_success:
                    self._tripped = True
                    self._reason = "已试 %d 张成功率仅 %d%%" % (
                        self._tried,
                        int(rate * 100),
                    )
                    return True
            return False


def ai_tag_memes(
    base_url,
    api_key,
    model,
    meme_list,
    on_progress=None,
    on_result=None,
    should_stop=None,
    style="general",
    concurrency=4,
    breaker=None,
    on_breaker=None,
):
    """并发批量标注表情包

    meme_list: [{"id": int, "path": str}, ...]
    on_progress(done, total): 每完成一张回调，done 单调递增
    on_result(item): 每完成一张回调原始结果，含 error 字段（便于打日志）
    breaker: 可选 CircuitBreaker，不传则本函数内部自建一个（隔离测试友好）
    on_breaker(reason): 触发熔断时回调一次，供上层中止同组任务
    返回 [{"id": int, "name": str, "ocr": str}, ...]，仅含成功条目。
    """
    results = []
    total = len(meme_list)
    if not total:
        return results
    style_text = _STYLE_NAMES.get(style, _STYLE_NAMES["general"])
    counter = [0]
    lock = threading.Lock()
    brk = breaker if breaker is not None else CircuitBreaker()
    # 熔断后立刻停：既让在途任务尽早收手，也阻止排队任务被提交
    stop_flag = threading.Event()

    def _abort():
        """是否应停止（用户取消 或 已熔断）"""
        if stop_flag.is_set():
            return True
        if should_stop and should_stop():
            return True
        return False

    def _trip(reason):
        with lock:
            first = not stop_flag.is_set()
            stop_flag.set()
        if first and on_breaker:
            try:
                on_breaker(reason)
            except Exception:
                pass

    def report(item):
        with lock:
            counter[0] += 1
            done = counter[0]
        if on_progress:
            try:
                on_progress(done, total)
            except Exception:
                pass
        if on_result:
            try:
                on_result(item)
            except Exception:
                pass

    def work(item):
        if _abort():
            return None
        data_uri = encode_image_base64(item["path"])
        if not data_uri:
            # 本地问题：计入统计但不算失败（不该拖垮熔断）
            return {
                "id": item["id"],
                "name": "",
                "ocr": "",
                "error": "无法读取图片",
                "local": True,
            }
        try:
            content = chat_completion(
                base_url,
                api_key,
                model,
                _build_messages(data_uri, style_text),
                should_stop=_abort,
            )
        except AIServiceError as e:
            if brk.record(False, e):
                _trip(brk.reason)
            return {
                "id": item["id"],
                "name": "",
                "ocr": "",
                "error": str(e),
                "kind": e.kind,
            }
        except Exception as e:
            if brk.record(False, None):
                _trip(brk.reason)
            return {"id": item["id"], "name": "", "ocr": "", "error": str(e)}

        parsed = _parse_ai_json(content)
        if isinstance(parsed, dict):
            name = clean_display_name(parsed.get("name", ""))
        else:
            name = ""
        if not name:
            # 内容问题：AI 活着，不算服务故障
            brk.record(True)
            return {
                "id": item["id"],
                "name": "",
                "ocr": "",
                "error": "AI 未返回有效名称",
                "kind": "content",
            }
        brk.record(True)
        return {
            "id": item["id"],
            "name": name,
            "ocr": clean_display_name(parsed.get("ocr", "")),
            "error": "",
        }

    def task(item):
        # 与 work 分离，保证无论成功/失败/异常都回调一次进度与结果
        try:
            r = work(item)
        except Exception as e:  # 兜底：work 内部已捕获，此处仅防御
            r = {"id": item.get("id"), "name": "", "ocr": "", "error": str(e)}
        report(r)
        return r

    workers = max(1, min(8, int(concurrency or 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(task, meme_list):
            if item is None:
                break
            # 仅成功条目进结果；失败条目已通过 on_result 上报，供调用方记日志
            if item.get("name"):
                results.append(item)
            elif brk.tripped:
                # 在途结果处理完后尽早退出，不再等 pool 把剩余任务跑完
                break
    return results
