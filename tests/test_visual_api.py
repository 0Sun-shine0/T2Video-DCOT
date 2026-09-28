"""即梦接口测试：签名交叉验证、限流、重试语义、缓存。"""

import datetime
import hashlib
import hmac
import json
import re

import pytest
import requests

from fakes import FakeResponse, FakeSession
from t2video import visual_api, settings
from t2video.visual_api import VisualApiClient, Throttle, canonical_query, signed_headers

STAMP = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
QUERY = "Action=CVProcess&Version=2022-08-31"
BODY = '{"prompt": "白日依山尽", "return_url": true}'


def reference_signature(secret, query, body, moment):
    """照官方样例的步骤独立重算，避免用被测代码验证自己。"""
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    datestamp = moment.strftime("%Y%m%d")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    headers = (
        "content-type:application/json\n"
        f"host:{settings.API_HOST}\n"
        f"x-content-sha256:{digest}\n"
        f"x-date:{stamp}\n"
    )
    canonical = "\n".join(["POST", "/", query, headers,
                           "content-type;host;x-content-sha256;x-date", digest])
    scope = f"{datestamp}/{settings.API_REGION}/{settings.API_SERVICE}/request"
    to_sign = "\n".join(["HMAC-SHA256", stamp, scope,
                         hashlib.sha256(canonical.encode("utf-8")).hexdigest()])

    def sign(key, message):
        if isinstance(key, str):
            key = key.encode("utf-8")
        return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

    key = sign(secret, datestamp)
    key = sign(key, settings.API_REGION)
    key = sign(key, settings.API_SERVICE)
    key = sign(key, "request")
    return hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def signature_in(headers):
    match = re.search(r"Signature=([0-9a-f]+)", headers["Authorization"])
    assert match
    return match.group(1)


class TestSigning:
    def test_matches_reference(self):
        headers = signed_headers("AKID", "SECRET", QUERY, BODY, moment=STAMP)
        assert signature_in(headers) == reference_signature("SECRET", QUERY, BODY, STAMP)

    def test_authorization_shape(self):
        headers = signed_headers("AKID", "SECRET", QUERY, BODY, moment=STAMP)
        assert headers["X-Date"] == "20260102T030405Z"
        assert "Credential=AKID/20260102/cn-north-1/cv/request" in headers["Authorization"]
        assert headers["Authorization"].startswith("HMAC-SHA256 ")

    def test_body_hash_utf8(self):
        headers = signed_headers("AKID", "SECRET", QUERY, BODY, moment=STAMP)
        assert headers["X-Content-Sha256"] == hashlib.sha256(BODY.encode("utf-8")).hexdigest()

    def test_body_change_alters_signature(self):
        first = signed_headers("AKID", "SECRET", QUERY, BODY, moment=STAMP)
        second = signed_headers("AKID", "SECRET", QUERY, BODY + " ", moment=STAMP)
        assert signature_in(first) != signature_in(second)

    def test_secret_change_alters_signature(self):
        first = signed_headers("AKID", "SECRET", QUERY, BODY, moment=STAMP)
        second = signed_headers("AKID", "OTHER", QUERY, BODY, moment=STAMP)
        assert signature_in(first) != signature_in(second)

    def test_query_sorted(self):
        assert canonical_query("CVProcess") == f"Action=CVProcess&Version={settings.API_VERSION}"


class TestThrottle:
    def test_allows_up_to_quota(self, sleeps):
        record, recorded = sleeps
        gate = Throttle(per_minute=3, sleep=record)
        for _ in range(3):
            gate.wait()
        assert recorded == []

    def test_blocks_beyond_quota(self, sleeps):
        record, recorded = sleeps
        gate = Throttle(per_minute=1, sleep=record)
        gate.wait()

        # 窗口内再取一次必然要等，等到窗口翻页后才放行；
        # 这里把窗口起点前移来模拟时间流逝，避免真的睡 60 秒。
        def tick(_):
            gate._since -= 61

        gate._sleep = tick
        gate.wait()
        assert gate._used == 1


def make_client(responses, sleep=lambda _: None):
    session = FakeSession(responses)
    return VisualApiClient(access="AKID", secret="SECRET", session=session, sleep=sleep), session


class TestCall:
    def test_success(self, sleeps):
        client, session = make_client([FakeResponse(200, {"data": {"task_id": "t"}})])
        assert client.call("CVProcess", {"a": 1}) == {"data": {"task_id": "t"}}
        assert len(session.calls) == 1

    def test_body_sent_as_utf8_bytes(self):
        client, session = make_client([FakeResponse(200, {})])
        client.call("CVProcess", {"prompt": "白日依山尽"})
        raw = session.calls[0][2]["data"]
        assert isinstance(raw, bytes)
        assert json.loads(raw.decode("utf-8"))["prompt"] == "白日依山尽"

    def test_signed_headers_attached(self):
        client, session = make_client([FakeResponse(200, {})])
        client.call("CVProcess", {"a": 1})
        assert {"X-Date", "Authorization", "X-Content-Sha256", "Content-Type"} <= set(
            session.calls[0][2]["headers"])

    def test_retries_on_429_with_backoff(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([FakeResponse(429, text="slow"),
                                       FakeResponse(429, text="slow"),
                                       FakeResponse(200, {"ok": True})], record)
        assert client.call("CVProcess", {"a": 1}) == {"ok": True}
        assert recorded == [2.0, 4.0]

    def test_gives_up_after_retries(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([FakeResponse(500, text="boom")] * 3, record)
        assert client.call("CVProcess", {"a": 1}, attempts=3) is None
        assert len(session.calls) == 3

    def test_client_error_no_retry(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([FakeResponse(403, text="nope")], record)
        assert client.call("CVProcess", {"a": 1}) is None
        assert len(session.calls) == 1 and recorded == []

    def test_network_error_retries(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([requests.ConnectionError("down")] * 3, record)
        assert client.call("CVProcess", {"a": 1}, attempts=3) is None
        assert recorded == [2.0, 4.0]

    def test_invalid_json(self):
        client, _ = make_client([FakeResponse(200, payload=None, text="nope")])
        assert client.call("CVProcess", {"a": 1}) is None

    def test_missing_credentials(self):
        client = VisualApiClient(access="", secret="", session=FakeSession([]))
        with pytest.raises(visual_api.MissingCredentials):
            client.call("CVProcess", {"a": 1})


class TestTextToImage:
    def test_success(self):
        client, session = make_client([FakeResponse(200, {"data": {"image_urls": ["http://i"]}})])
        assert client.text_to_image("落日") == "http://i"
        assert json.loads(session.calls[0][2]["data"])["req_key"] == settings.TEXT2IMAGE_REQ_KEY

    def test_no_url_returns_none(self):
        client, _ = make_client([FakeResponse(200, {"data": {"image_urls": []}})])
        assert client.text_to_image("落日") is None

    def test_cache_hit_skips_request(self):
        client, session = make_client([FakeResponse(200, {"data": {"image_urls": ["http://i"]}})])
        client.text_to_image("落日")
        client.text_to_image("落日")
        assert len(session.calls) == 1

    def test_cache_key_includes_reference(self):
        client, session = make_client([
            FakeResponse(200, {"data": {"image_urls": ["a"]}}),
            FakeResponse(200, {"data": {"image_urls": ["b"]}}),
        ])
        client.text_to_image("落日")
        client.text_to_image("落日", reference=["http://ref"])
        assert len(session.calls) == 2


class TestSubmit:
    def test_success(self):
        client, session = make_client([FakeResponse(200, {"data": {"task_id": "t1"}})])
        assert client._submit("http://i") == "t1"
        assert json.loads(session.calls[0][2]["data"])["aspect_ratio"] == settings.VIDEO_ASPECT_RATIO

    def test_concurrency_retry_then_success(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([FakeResponse(200, {"code": visual_api.CONCURRENCY_CODE}),
                                       FakeResponse(200, {"data": {"task_id": "t1"}})], record)
        assert client._submit("http://i") == "t1"
        assert recorded and recorded[0] == settings.CONCURRENCY_BACKOFF

    def test_concurrency_exhausted(self, sleeps):
        record, recorded = sleeps
        responses = [FakeResponse(200, {"code": visual_api.CONCURRENCY_CODE})] * settings.CONCURRENCY_RETRY
        client, session = make_client(responses, record)
        assert client._submit("http://i") is None
        assert len(session.calls) == settings.CONCURRENCY_RETRY

    def test_other_error_no_retry(self):
        client, session = make_client([FakeResponse(200, {"code": 400})])
        assert client._submit("http://i") is None
        assert len(session.calls) == 1


class TestPoll:
    def test_done(self):
        client, _ = make_client([FakeResponse(200, {"data": {"status": "done", "video_url": "v"}})])
        assert client._poll("t") == "v"

    def test_done_without_url(self):
        client, _ = make_client([FakeResponse(200, {"data": {"status": "done"}})])
        assert client._poll("t") is None

    def test_failed(self):
        client, _ = make_client([FakeResponse(200, {"data": {"status": "failed", "message": "x"}})])
        assert client._poll("t") is None

    def test_canceled(self):
        client, _ = make_client([FakeResponse(200, {"data": {"status": "canceled"}})])
        assert client._poll("t") is None

    def test_polls_until_done(self, sleeps):
        record, recorded = sleeps
        client, session = make_client([
            FakeResponse(200, {"data": {"status": "processing"}}),
            FakeResponse(200, {"data": {"status": "processing"}}),
            FakeResponse(200, {"data": {"status": "done", "video_url": "v"}}),
        ], record)
        assert client._poll("t") == "v"
        assert len(recorded) == 2

    def test_backoff_grows_and_caps(self, sleeps, monkeypatch):
        record, recorded = sleeps
        monkeypatch.setattr(settings, "POLL_INTERVAL", 10.0)
        monkeypatch.setattr(settings, "POLL_MAX_INTERVAL", 25.0)
        monkeypatch.setattr(settings, "POLL_MAX_ATTEMPTS", 4)
        client, _ = make_client([FakeResponse(200, {"data": {"status": "processing"}})] * 4, record)
        assert client._poll("t") is None
        assert recorded == [15.0, 22.5, 25.0]

    def test_transient_failure_does_not_abort(self):
        client, _ = make_client([
            FakeResponse(500, text="boom"),
            FakeResponse(500, text="boom"),
            FakeResponse(500, text="boom"),
            FakeResponse(200, {"data": {"status": "done", "video_url": "v"}}),
        ])
        assert client._poll("t") == "v"


class TestImageToVideo:
    def test_success(self):
        client, _ = make_client([
            FakeResponse(200, {"data": {"task_id": "t"}}),
            FakeResponse(200, {"data": {"status": "done", "video_url": "v"}}),
        ])
        assert client.image_to_video("http://i") == "v"

    def test_cache_short_circuits(self):
        client, session = make_client([
            FakeResponse(200, {"data": {"task_id": "t"}}),
            FakeResponse(200, {"data": {"status": "done", "video_url": "v"}}),
        ])
        client.image_to_video("http://i")
        client.image_to_video("http://i")
        assert len(session.calls) == 2

    def test_submit_failure(self):
        client, _ = make_client([FakeResponse(200, {"code": 400})])
        assert client.image_to_video("http://i") is None
