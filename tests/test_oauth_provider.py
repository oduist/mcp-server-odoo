"""Tests for the in-memory OAuth provider."""

import time

import pytest

from mcp.shared.auth import OAuthClientInformationFull

from mcp_server_odoo.oauth_provider import (
    ACCESS_TOKEN_TTL,
    OdooOAuthProvider,
)

SERVER_URL = "https://mcp.example.com"
AUTH_TOKEN = "secret-123"


def _make_provider():
    return OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN)


def _make_client(client_id="test-client"):
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="test-secret",
        redirect_uris=["https://app.example.com/callback"],
    )


# ── Client registration ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_and_get_client():
    p = _make_provider()
    client = _make_client()
    await p.register_client(client)

    result = await p.get_client("test-client")
    assert result is not None
    assert result.client_id == "test-client"


@pytest.mark.asyncio
async def test_get_unknown_client_returns_none():
    p = _make_provider()
    assert await p.get_client("nonexistent") is None


# ── Authorization flow ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_authorize_returns_consent_url():
    from mcp.server.auth.provider import AuthorizationParams

    p = _make_provider()
    client = _make_client()
    params = AuthorizationParams(
        state="abc",
        scopes=["odoo"],
        code_challenge="challenge123",
        redirect_uri="https://app.example.com/callback",
        redirect_uri_provided_explicitly=True,
    )

    url = await p.authorize(client, params)
    assert url.startswith(f"{SERVER_URL}/oauth/consent?")
    assert "request_id=" in url
    assert "client_id=test-client" in url


@pytest.mark.asyncio
async def test_complete_authorization_and_exchange():
    from mcp.server.auth.provider import AuthorizationParams

    p = _make_provider()
    client = _make_client()
    params = AuthorizationParams(
        state="xyz",
        scopes=["odoo"],
        code_challenge="challenge123",
        redirect_uri="https://app.example.com/callback",
        redirect_uri_provided_explicitly=True,
    )

    consent_url = await p.authorize(client, params)
    # Extract request_id from URL
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(consent_url).query)
    request_id = qs["request_id"][0]

    # Complete authorization
    redirect_url = p.complete_authorization(request_id, "test-client")
    assert "code=" in redirect_url
    assert "state=xyz" in redirect_url

    # Extract code
    redirect_qs = parse_qs(urlparse(redirect_url).query)
    code = redirect_qs["code"][0]

    # Load and exchange code
    auth_code = await p.load_authorization_code(client, code)
    assert auth_code is not None
    assert auth_code.client_id == "test-client"

    token = await p.exchange_authorization_code(client, auth_code)
    assert token.access_token
    assert token.refresh_token
    assert token.token_type == "Bearer"
    assert token.expires_in == ACCESS_TOKEN_TTL

    # Code is consumed (single-use)
    assert await p.load_authorization_code(client, code) is None


# ── Token verification ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_access_token():
    from mcp.server.auth.provider import AuthorizationParams

    p = _make_provider()
    client = _make_client()
    params = AuthorizationParams(
        state="s",
        scopes=[],
        code_challenge="c",
        redirect_uri="https://app.example.com/callback",
        redirect_uri_provided_explicitly=True,
    )
    url = await p.authorize(client, params)
    from urllib.parse import parse_qs, urlparse

    rid = parse_qs(urlparse(url).query)["request_id"][0]
    rurl = p.complete_authorization(rid, "test-client")
    code = parse_qs(urlparse(rurl).query)["code"][0]
    auth_code = await p.load_authorization_code(client, code)
    token = await p.exchange_authorization_code(client, auth_code)

    at = await p.load_access_token(token.access_token)
    assert at is not None
    assert at.client_id == "test-client"


@pytest.mark.asyncio
async def test_expired_access_token_returns_none():
    p = _make_provider()
    from mcp_server_odoo.oauth_provider import AccessToken

    p._access_tokens["expired"] = AccessToken(
        token="expired",
        client_id="c",
        scopes=[],
        expires_at=int(time.time()) - 10,
    )
    assert await p.load_access_token("expired") is None


# ── Refresh tokens ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_token_rotation():
    from mcp.server.auth.provider import AuthorizationParams

    p = _make_provider()
    client = _make_client()
    params = AuthorizationParams(
        state="s",
        scopes=["odoo"],
        code_challenge="c",
        redirect_uri="https://app.example.com/callback",
        redirect_uri_provided_explicitly=True,
    )
    url = await p.authorize(client, params)
    from urllib.parse import parse_qs, urlparse

    rid = parse_qs(urlparse(url).query)["request_id"][0]
    rurl = p.complete_authorization(rid, "test-client")
    code = parse_qs(urlparse(rurl).query)["code"][0]
    auth_code = await p.load_authorization_code(client, code)
    token = await p.exchange_authorization_code(client, auth_code)

    # Refresh
    rt = await p.load_refresh_token(client, token.refresh_token)
    assert rt is not None

    new_token = await p.exchange_refresh_token(client, rt, ["odoo"])
    assert new_token.access_token != token.access_token
    assert new_token.refresh_token != token.refresh_token

    # Old refresh token is consumed
    assert await p.load_refresh_token(client, token.refresh_token) is None


# ── Revocation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_access_token():
    p = _make_provider()
    from mcp_server_odoo.oauth_provider import AccessToken

    at = AccessToken(token="t1", client_id="c", scopes=[], expires_at=int(time.time()) + 3600)
    p._access_tokens["t1"] = at

    await p.revoke_token(at)
    assert await p.load_access_token("t1") is None
