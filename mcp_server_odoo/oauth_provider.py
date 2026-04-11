"""In-memory OAuth 2.0 authorization server for MCP HTTP transport.

Implements the ``OAuthAuthorizationServerProvider`` protocol expected by
the MCP Python SDK so that OAuth-only clients (e.g. Claude.ai) can
connect to the server.

The authorization "gate" is the static ``ODOO_MCP_AUTH_TOKEN`` — the user
must enter it on a minimal consent page to prove they are allowed to use
this server.  Everything else (clients, codes, tokens) lives in memory
for the lifetime of the process.
"""

import hashlib
import logging
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)

# Token lifetime constants
AUTH_CODE_TTL = 300  # 5 minutes
ACCESS_TOKEN_TTL = 3600  # 1 hour
REFRESH_TOKEN_TTL = 86400 * 30  # 30 days


class OdooOAuthProvider:
    """Minimal in-memory OAuth provider gated by a static secret token."""

    def __init__(self, server_url: str, auth_token: str):
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token

        # In-memory stores keyed by their respective identifiers
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}

        # Map pending authorization requests so the consent page can
        # look them up after the user submits the form.
        self._pending_auth: dict[str, AuthorizationParams] = {}

    # ── Client registration ─────────────────────────────────────────

    async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info
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

    def complete_authorization(
        self, request_id: str, client_id: str
    ) -> str:
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
        logger.info("OAuth: revoked token")
