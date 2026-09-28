"""视觉生成服务接口封装：V4 签名、限流、文生图、图生视频。

本模块对接第三方视觉生成服务（火山引擎视觉服务 / 即梦），接口提供方信息
在此如实写明；模块名按自身职责命名，不使用第三方产品名。

重试分两层且互不叠加：签名请求层只对 429/5xx 做退避重试；
业务层只在"并发超限"这个可恢复错误上重试。两层的次数不相乘。
"""

import datetime
import hashlib
import hmac
import json
import logging
import threading
import time

import requests

from . import settings

LOGGER = logging.getLogger(__name__)

ALGORITHM = "HMAC-SHA256"
SIGNED_HEADERS = "content-type;host;x-content-sha256;x-date"

CONCURRENCY_CODE = 50430

# 即梦文生图接口在当前调用方式下不接受参考图参数。若控制台确认了字段名
# （image_urls / binary_data_base64 之类），填到这里即可启用首尾帧连贯；
# 在此之前跨镜连贯由 prompt_builder 的承接从句负责，不做假的参数填充。
REFERENCE_IMAGE_FIELD = None

# 图生视频同样只有 image_urls + aspect_ratio。运镜提示词在没有确认字段名之前
# 不会进请求体，此时它只作为分镜规格参与验证与修正（见 gui._render_clip）。
IMAGE2VIDEO_PROMPT_FIELD = None


class MissingCredentials(RuntimeError):
    """密钥缺失。与网络失败区分开：这种情况重试没有意义，直接向上报。"""


class ApiUnavailable(RuntimeError):
    """可重试的服务端错误，对应 429 与 5xx。"""

    def __init__(self, status, detail=""):
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail[:200]}")


def _hmac(key, message):
    """对消息做一次 HMAC-SHA256，返回原始字节。"""
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret, datestamp):
    """逐级派生签名密钥：日期 → 区域 → 服务 → request。

    派生链是规范规定的，任一级顺序错误都会得到无效签名。
    """
    key = _hmac(secret.encode("utf-8"), datestamp)
    key = hmac.new(key, settings.API_REGION.encode("utf-8"), hashlib.sha256).digest()
    key = hmac.new(key, settings.API_SERVICE.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(key, b"request", hashlib.sha256).digest()


def canonical_query(action):
    """构造查询串。参数按字典序排列，否则签名与请求不一致。"""
    return f"Action={action}&Version={settings.API_VERSION}"


def signed_headers(access, secret, query, body, moment=None):
    """生成 V4 签名头。moment 可注入，方便用固定时间戳做回归测试。"""
    moment = moment or datetime.datetime.now(datetime.timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    datestamp = moment.strftime("%Y%m%d")

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    # 末尾的换行不能省：规范要求在 canonical headers 之后留一个空行
    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{settings.API_HOST}\n"
        f"x-content-sha256:{digest}\n"
        f"x-date:{stamp}\n"
    )
    canonical_request = "\n".join([
        "POST", "/", query, canonical_headers, SIGNED_HEADERS, digest,
    ])
    scope = f"{datestamp}/{settings.API_REGION}/{settings.API_SERVICE}/request"
    to_sign = "\n".join([
        ALGORITHM,
        stamp,
        scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signature = hmac.new(signing_key(secret, datestamp), to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()

    return {
        "X-Date": stamp,
        "Authorization": (
            f"{ALGORITHM} Credential={access}/{scope}, "
            f"SignedHeaders={SIGNED_HEADERS}, Signature={signature}"
        ),
        "X-Content-Sha256": digest,
        "Content-Type": "application/json",
    }


class Throttle:
    """每分钟请求数上限。窗口内超了就等到窗口结束，不做排队堆积。"""

    def __init__(self, per_minute=None, sleep=time.sleep):
        self.per_minute = per_minute or settings.REQUESTS_PER_MINUTE
        self._sleep = sleep
        self._lock = threading.Lock()
        self._used = 0
        self._since = time.monotonic()

    def wait(self):
        """取一个请求名额；窗口已满则阻塞到窗口结束再取。"""
        while True:
            with self._lock:
                elapsed = time.monotonic() - self._since
                if elapsed >= 60:
                    self._used = 0
                    self._since = time.monotonic()
                    elapsed = 0.0
                if self._used < self.per_minute:
                    self._used += 1
                    return
                remaining = 60 - elapsed
            LOGGER.info("已达每分钟请求上限，等待 %.1fs", remaining)
            self._sleep(max(remaining, 0.0))


class VisualApiClient:
    """视觉生成服务客户端。

    持有会话级缓存：同一提示词不重复出图，同一输入图不重复出视频。
    两者都以内容为键，因此缓存命中与否不受分镜顺序影响。
    """

    def __init__(self, access=None, secret=None, session=None, sleep=time.sleep):
        self.access = settings.ACCESS_KEY if access is None else access
        self.secret = settings.SECRET_KEY if secret is None else secret
        self.session = session or requests.Session()
        self.throttle = Throttle(sleep=sleep)
        self._sleep = sleep
        self.image_cache = {}
        self.video_cache = {}

    def call(self, action, body, attempts=None):
        """调用一个 Action。可重试错误退避重试，其余返回 None。"""
        if not self.access or not self.secret:
            raise MissingCredentials(settings.CREDENTIAL_HINT)

        attempts = attempts or settings.REQUEST_RETRY
        query = canonical_query(action)
        payload = json.dumps(body, ensure_ascii=False)

        for attempt in range(1, attempts + 1):
            try:
                return self._post(action, query, payload)
            except ApiUnavailable as exc:
                if attempt == attempts:
                    LOGGER.error("%s 重试 %d 次后仍失败: %s", action, attempts, exc)
                    return None
                delay = settings.REQUEST_BACKOFF * (2 ** (attempt - 1))
                LOGGER.warning("%s 第 %d/%d 次失败(%s)，%.1fs 后重试",
                               action, attempt, attempts, exc.status, delay)
                self._sleep(delay)
            except requests.exceptions.RequestException as exc:
                if attempt == attempts:
                    LOGGER.error("%s 网络异常: %s", action, exc)
                    return None
                delay = settings.REQUEST_BACKOFF * (2 ** (attempt - 1))
                LOGGER.warning("%s 第 %d/%d 次网络异常: %s，%.1fs 后重试",
                               action, attempt, attempts, exc, delay)
                self._sleep(delay)
        return None

    def _post(self, action, query, payload):
        """发一次请求。把 429 与 5xx 转成可重试异常，4xx 直接记日志返回 None。"""
        self.throttle.wait()
        response = self.session.post(
            f"{settings.API_ENDPOINT}?{query}",
            headers=signed_headers(self.access, self.secret, query, payload),
            data=payload.encode("utf-8"),
            timeout=settings.HTTP_TIMEOUT,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise ApiUnavailable(response.status_code, response.text)
        if response.status_code >= 400:
            LOGGER.error("%s 返回 %s: %s", action, response.status_code, response.text[:300])
            return None
        try:
            return response.json()
        except ValueError:
            LOGGER.error("%s 响应不是合法 JSON", action)
            return None

    def text_to_image(self, prompt, reference=None):
        """生成一张图，返回图片 URL。相同提示词命中缓存，不重复计费。"""
        cache_key = (prompt, tuple(reference or ()))
        if cache_key in self.image_cache:
            LOGGER.info("命中图像缓存")
            return self.image_cache[cache_key]

        body = {
            "req_key": settings.TEXT2IMAGE_REQ_KEY,
            "prompt": prompt,
            "return_url": True,
        }
        if reference and REFERENCE_IMAGE_FIELD:
            body[REFERENCE_IMAGE_FIELD] = list(reference)

        data = (self.call("CVProcess", body) or {}).get("data") or {}
        urls = data.get("image_urls") or []
        if not urls:
            LOGGER.error("文生图未返回图片")
            return None

        self.image_cache[cache_key] = urls[0]
        return urls[0]

    def image_to_video(self, image_url, motion_prompt=None):
        """图生视频。同一张输入图只提交一次任务。

        motion_prompt 只有在 IMAGE2VIDEO_PROMPT_FIELD 配置后才会进请求体。
        """
        if image_url in self.video_cache:
            LOGGER.info("命中视频缓存")
            return self.video_cache[image_url]

        task_id = self._submit(image_url, motion_prompt)
        if not task_id:
            return None
        video_url = self._poll(task_id)
        if video_url:
            self.video_cache[image_url] = video_url
        return video_url

    def _submit(self, image_url, motion_prompt=None):
        """提交图生视频任务，返回 task_id。

        只对并发超限这一个错误码做重试：它属于"稍后再试即可"的可恢复状态，
        其余错误码重试也不会成功。与请求层的退避重试相互独立，次数不相乘。
        """
        body = {
            "req_key": settings.IMAGE2VIDEO_REQ_KEY,
            "image_urls": [image_url],
            "aspect_ratio": settings.VIDEO_ASPECT_RATIO,
        }
        if motion_prompt and IMAGE2VIDEO_PROMPT_FIELD:
            body[IMAGE2VIDEO_PROMPT_FIELD] = motion_prompt
        for attempt in range(1, settings.CONCURRENCY_RETRY + 1):
            response = self.call("CVSync2AsyncSubmitTask", body)
            task_id = ((response or {}).get("data") or {}).get("task_id")
            if task_id:
                LOGGER.info("视频任务已提交: %s", task_id)
                return task_id

            if (response or {}).get("code") == CONCURRENCY_CODE:
                if attempt == settings.CONCURRENCY_RETRY:
                    LOGGER.error("并发超限重试 %d 次仍未提交成功", attempt)
                    return None
                delay = settings.CONCURRENCY_BACKOFF * attempt
                LOGGER.warning("并发超限，%.0fs 后重试 (%d/%d)",
                               delay, attempt, settings.CONCURRENCY_RETRY)
                self._sleep(delay)
                continue

            LOGGER.error("视频任务提交失败: %s", str(response)[:200])
            return None
        return None

    def _poll(self, task_id):
        """轮询任务直到出结果。

        间隔按倍数递增并设上限：视频生成通常要一到数分钟，
        固定短间隔会白白消耗请求额度，固定长间隔又会让快任务等待过久。
        查询本身失败不中止轮询，只是把下次间隔拉长一倍。
        """
        body = {"req_key": settings.IMAGE2VIDEO_REQ_KEY, "task_id": task_id}
        interval = settings.POLL_INTERVAL

        for attempt in range(1, settings.POLL_MAX_ATTEMPTS + 1):
            response = self.call("CVSync2AsyncGetResult", body)

            if not response:
                interval = min(interval * 2, settings.POLL_MAX_INTERVAL)
            else:
                data = response.get("data") or {}
                status = data.get("status", "")
                if status == "done":
                    url = data.get("video_url", "")
                    if url:
                        return url
                    LOGGER.error("任务完成但没有视频地址")
                    return None
                if status in ("failed", "canceled"):
                    LOGGER.error("任务%s：%s", "失败" if status == "failed" else "被取消",
                                 data.get("message", "无详细信息"))
                    return None
                if status in ("processing", "pending", "in_queue"):
                    interval = min(interval * settings.POLL_GROWTH, settings.POLL_MAX_INTERVAL)

            if attempt < settings.POLL_MAX_ATTEMPTS:
                self._sleep(interval)

        LOGGER.error("任务轮询超过 %d 次仍未完成", settings.POLL_MAX_ATTEMPTS)
        return None
