"""Bearer token authentication middleware for HTTP transport.

Validates that every incoming HTTP request carries a valid
``Authorization: Bearer <token>`` header before forwarding it
to the MCP streamable-HTTP handler.
"""

import hmac
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Paths that bypass authentication (e.g. health check)
_PUBLIC_PATHS = frozenset({"/health"})


class BearerTokenMiddleware:
    """ASGI middleware that enforces a static Bearer token on every request."""

    def __init__(self, app: Any, *, token: str):
        self.app = app
        self._token = token

    async def __call__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        if not auth_value.startswith("Bearer "):
            await self._send_error(send, 401, "Missing or malformed Authorization header")
            return

        presented = auth_value[7:]  # strip "Bearer "
        if not hmac.compare_digest(presented, self._token):
            logger.warning("HTTP auth: invalid bearer token")
            await self._send_error(send, 401, "Invalid bearer token")
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_error(send: Callable, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
