"""Tests for HTTP Bearer token authentication middleware."""

import json

import pytest

from mcp_server_odoo.http_auth import BearerTokenMiddleware

TOKEN = "test-secret-token-123"


# ── helpers ──────────────────────────────────────────────────────────────────


class _CapturedResponse:
    """Captures ASGI send() calls so we can inspect the response."""

    def __init__(self):
        self.status = None
        self.headers = {}
        self.body = b""

    async def send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = dict(message.get("headers", []))
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")


class _DummyApp:
    """Minimal ASGI app that records whether it was called."""

    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        if send is None:
            return
        body = b'{"ok":true}'
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _http_scope(path="/mcp", auth_header=None):
    headers = []
    if auth_header is not None:
        headers.append((b"authorization", auth_header.encode()))
    return {
        "type": "http",
        "path": path,
        "headers": headers,
    }


# ── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_token_passes_through():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)
    resp = _CapturedResponse()

    await mw(_http_scope(auth_header=f"Bearer {TOKEN}"), None, resp.send)

    assert inner.called
    assert resp.status == 200


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)
    resp = _CapturedResponse()

    await mw(_http_scope(), None, resp.send)

    assert not inner.called
    assert resp.status == 401
    body = json.loads(resp.body)
    assert "Missing" in body["error"]


@pytest.mark.asyncio
async def test_wrong_token_returns_401():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)
    resp = _CapturedResponse()

    await mw(_http_scope(auth_header="Bearer wrong-token"), None, resp.send)

    assert not inner.called
    assert resp.status == 401
    body = json.loads(resp.body)
    assert "Invalid" in body["error"]


@pytest.mark.asyncio
async def test_non_bearer_scheme_returns_401():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)
    resp = _CapturedResponse()

    await mw(_http_scope(auth_header="Basic dXNlcjpwYXNz"), None, resp.send)

    assert not inner.called
    assert resp.status == 401


@pytest.mark.asyncio
async def test_health_endpoint_bypasses_auth():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)
    resp = _CapturedResponse()

    await mw(_http_scope(path="/health"), None, resp.send)

    assert inner.called
    assert resp.status == 200


@pytest.mark.asyncio
async def test_non_http_scope_passes_through():
    inner = _DummyApp()
    mw = BearerTokenMiddleware(inner, token=TOKEN)

    await mw({"type": "lifespan"}, None, None)

    assert inner.called
