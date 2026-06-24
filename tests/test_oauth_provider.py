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


# ── Persistence ─────────────────────────────────────────────────────


async def _mint_tokens(p):
    """Run the full authorize→consent→exchange flow, returning (client, token)."""
    from urllib.parse import parse_qs, urlparse

    from mcp.server.auth.provider import AuthorizationParams

    client = _make_client()
    await p.register_client(client)
    params = AuthorizationParams(
        state="s",
        scopes=["odoo"],
        code_challenge="c",
        redirect_uri="https://app.example.com/callback",
        redirect_uri_provided_explicitly=True,
    )
    url = await p.authorize(client, params)
    rid = parse_qs(urlparse(url).query)["request_id"][0]
    rurl = p.complete_authorization(rid, client.client_id)
    code = parse_qs(urlparse(rurl).query)["code"][0]
    auth_code = await p.load_authorization_code(client, code)
    return client, await p.exchange_authorization_code(client, auth_code)


@pytest.mark.asyncio
async def test_state_persisted_and_reloaded(tmp_path):
    store = tmp_path / "oauth_state.json"
    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    client, token = await _mint_tokens(p)
    assert store.exists()

    # A fresh provider over the same path restores client + tokens —
    # i.e. the client does NOT have to log in again after a restart.
    p2 = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    assert await p2.get_client(client.client_id) is not None
    assert await p2.load_access_token(token.access_token) is not None
    rt = await p2.load_refresh_token(client, token.refresh_token)
    assert rt is not None
    # The restored refresh token still works.
    new_token = await p2.exchange_refresh_token(client, rt, ["odoo"])
    assert new_token.access_token


@pytest.mark.asyncio
async def test_state_file_has_restrictive_permissions(tmp_path):
    import stat

    store = tmp_path / "nested" / "oauth_state.json"
    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    await _mint_tokens(p)

    assert store.exists()  # parent dir was created
    assert stat.S_IMODE(store.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_expired_tokens_pruned_on_load(tmp_path):
    import json

    past = int(time.time()) - 10
    store = tmp_path / "oauth_state.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "clients": {},
                "access_tokens": {
                    "old": {"token": "old", "client_id": "c", "scopes": [], "expires_at": past}
                },
                "refresh_tokens": {
                    "old-r": {"token": "old-r", "client_id": "c", "scopes": [], "expires_at": past}
                },
            }
        )
    )

    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    assert await p.load_access_token("old") is None
    assert "old-r" not in p._refresh_tokens


@pytest.mark.asyncio
async def test_corrupt_state_file_starts_empty(tmp_path):
    store = tmp_path / "oauth_state.json"
    store.write_text("{ not valid json ]")

    # Must not raise — a broken file just starts fresh.
    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    assert await p.get_client("anything") is None


@pytest.mark.asyncio
async def test_malformed_client_entry_ignored(tmp_path):
    import json

    store = tmp_path / "oauth_state.json"
    store.write_text(json.dumps({"version": 1, "clients": {"bad": {"not": "a client"}}}))

    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN, store_path=str(store))
    assert await p.get_client("bad") is None


@pytest.mark.asyncio
async def test_no_store_path_is_memory_only(tmp_path):
    p = OdooOAuthProvider(server_url=SERVER_URL, auth_token=AUTH_TOKEN)
    assert p._store_path is None

    _, token = await _mint_tokens(p)  # works fine without persistence
    assert await p.load_access_token(token.access_token) is not None
    assert list(tmp_path.iterdir()) == []  # provider wrote no files
