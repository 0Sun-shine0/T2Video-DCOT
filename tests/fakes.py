"""伪造的 HTTP 层，测试里绝不发真实请求。"""

import json

import requests


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", chunks=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")
        self.headers = {"content-length": str(sum(len(c) for c in (chunks or [])))}
        self._chunks = chunks or []

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


class FakeSession:
    """按顺序返回预设响应，并记录每次调用的参数。"""

    def __init__(self, responses=None, exception=None):
        self.responses = list(responses or [])
        self.exception = exception
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._next()

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._next()

    def _next(self):
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            raise AssertionError("FakeSession 没有更多预设响应")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last_body(self):
        return json.loads(self.calls[-1][2]["data"].decode("utf-8"))
