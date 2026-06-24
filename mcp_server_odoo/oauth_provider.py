"""OAuth 2.0 authorization server for MCP HTTP transport.

Implements the ``OAuthAuthorizationServerProvider`` protocol expected by
the MCP Python SDK so that OAuth-only clients (e.g. Claude.ai) can
connect to the server.

The authorization "gate" is the static ``ODOO_MCP_AUTH_TOKEN`` — the user
must enter it on a minimal consent page to prove they are allowed to use
this server.  Authorization codes and pending-consent state always live in
memory (they are short-lived, in-flight-only data), but registered clients
and access/refresh tokens are persisted to ``store_path`` when configured,
so client logins survive a process or container restart.  Without a store
path everything lives in memory for the lifetime of the process.
"""

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Token lifetime constants
AUTH_CODE_TTL = 300  # 5 minutes
ACCESS_TOKEN_TTL = 3600  # 1 hour
REFRESH_TOKEN_TTL = 86400 * 30  # 30 days


class OdooOAuthProvider:
    """Minimal OAuth provider gated by a static secret token.

    Clients and tokens are kept in memory and, when *store_path* is set,
    mirrored to a JSON file so they survive a restart.
    """

    def __init__(self, server_url: str, auth_token: str, store_path: Optional[str] = None):
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token

        # Optional on-disk mirror of clients + tokens (None = memory only).
        self._store_path: Optional[Path] = Path(store_path).expanduser() if store_path else None

        # In-memory stores keyed by their respective identifiers
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}

        # Map pending authorization requests so the consent page can
        # look them up after the user submits the form.
        self._pending_auth: dict[str, AuthorizationParams] = {}

        if self._store_path:
            self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Restore persisted clients and tokens from disk (best-effort).

        Auth codes and pending-consent state are intentionally not
        persisted: they are short-lived and only valid mid-handshake.
        A missing, unreadable, or malformed file simply starts empty —
        it must never crash startup.
        """
        path = self._store_path
        if not path or not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("OAuth: could not read state file %s: %s", path, exc)
            return

        now = int(time.time())

        def _live(raw: dict) -> bool:
            exp = raw.get("expires_at")
            return exp is None or exp > now

        try:
            for cid, raw in (data.get("clients") or {}).items():
                self._clients[cid] = OAuthClientInformationFull.model_validate(raw)
            for tok, raw in (data.get("access_tokens") or {}).items():
                if _live(raw):
                    self._access_tokens[tok] = AccessToken.model_validate(raw)
            for tok, raw in (data.get("refresh_tokens") or {}).items():
                if _live(raw):
                    self._refresh_tokens[tok] = RefreshToken.model_validate(raw)
        except (ValidationError, AttributeError) as exc:
            logger.warning("OAuth: ignoring malformed state in %s: %s", path, exc)
            self._clients.clear()
            self._access_tokens.clear()
            self._refresh_tokens.clear()
            return

        logger.info(
            "OAuth: restored %d client(s) and %d access + %d refresh token(s) from %s",
            len(self._clients),
            len(self._access_tokens),
            len(self._refresh_tokens),
            path,
        )

    def _save(self) -> None:
        """Persist clients and tokens to disk atomically (best-effort).

        Called after every mutation. Only currently-live tokens are
        written, so expired entries are pruned on the next save. The file
        holds bearer/refresh tokens, so it is created ``0600`` in a
        ``0700`` directory. A disk error is logged, never raised — it must
        not break a live OAuth request.
        """
        path = self._store_path
        if not path:
            return

        now = int(time.time())
        payload = {
            "version": 1,
            "clients": {
                cid: client.model_dump(mode="json") for cid, client in self._clients.items()
            },
            "access_tokens": {
                tok: at.model_dump(mode="json")
                for tok, at in self._access_tokens.items()
                if at.expires_at is None or at.expires_at > now
            },
            "refresh_tokens": {
                tok: rt.model_dump(mode="json")
                for tok, rt in self._refresh_tokens.items()
                if rt.expires_at is None or rt.expires_at > now
            },
        }

        tmp = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass  # best-effort; e.g. unsupported on the platform
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("OAuth: could not persist state to %s: %s", path, exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    # ── Client registration ─────────────────────────────────────────

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info
        self._save()
        logger.info(f"OAuth: registered client {client_info.client_id}")

    # ── Authorization ────────────────────────────────────────────────

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Return the URL of our consent page.

        The SDK's ``/authorize`` handler will redirect the user-agent here.
        """
        # Store params so the consent handler can retrieve them later
        request_id = secrets.token_urlsafe(32)
        self._pending_auth[request_id] = params

        query = urlencode(
            {
                "request_id": request_id,
                "client_id": client.client_id,
            }
        )
        return f"{self.server_url}/oauth/consent?{query}"

    # ── Consent completion (called by our custom route handler) ──────

    def complete_authorization(self, request_id: str, client_id: str) -> str:
        """Validate the pending request and return a redirect URL with the
        authorization code appended.

        Returns:
            Redirect URL to send the user-agent to.

        Raises:
            KeyError: if *request_id* is unknown.
        """
        params = self._pending_auth.pop(request_id)

        code = secrets.token_urlsafe(48)  # ≥ 256 bits
        now = time.time()

        self._auth_codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=now + AUTH_CODE_TTL,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )

        logger.info(f"OAuth: issued authorization code for client {client_id}")

        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    # ── Token exchange ───────────────────────────────────────────────

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> Optional[AuthorizationCode]:
        return self._auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        # Consume code (single-use)
        self._auth_codes.pop(authorization_code.code, None)

        now = int(time.time())
        access = secrets.token_urlsafe(48)
        refresh = secrets.token_urlsafe(48)

        self._access_tokens[access] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + ACCESS_TOKEN_TTL,
            resource=authorization_code.resource,
        )
        self._refresh_tokens[refresh] = RefreshToken(
            token=refresh,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
        )
        self._save()

        logger.info(f"OAuth: issued access token for client {client.client_id}")

        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else None,
            refresh_token=refresh,
        )

    # ── Refresh tokens ───────────────────────────────────────────────

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> Optional[RefreshToken]:
        return self._refresh_tokens.get(refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Revoke old refresh token (rotation)
        self._refresh_tokens.pop(refresh_token.token, None)

        now = int(time.time())
        new_access = secrets.token_urlsafe(48)
        new_refresh = secrets.token_urlsafe(48)
        effective_scopes = scopes or refresh_token.scopes

        self._access_tokens[new_access] = AccessToken(
            token=new_access,
            client_id=client.client_id,
            scopes=effective_scopes,
            expires_at=now + ACCESS_TOKEN_TTL,
        )
        self._refresh_tokens[new_refresh] = RefreshToken(
            token=new_refresh,
            client_id=client.client_id,
            scopes=effective_scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
        )
        self._save()

        logger.info(f"OAuth: refreshed token for client {client.client_id}")

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=" ".join(effective_scopes) if effective_scopes else None,
            refresh_token=new_refresh,
        )

    # ── Access token verification ────────────────────────────────────

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        at = self._access_tokens.get(token)
        if at and at.expires_at and at.expires_at < int(time.time()):
            self._access_tokens.pop(token, None)
            return None
        return at

    # ── Revocation ───────────────────────────────────────────────────

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
        elif isinstance(token, RefreshToken):
            self._refresh_tokens.pop(token.token, None)
        self._save()
        logger.info("OAuth: revoked token")
