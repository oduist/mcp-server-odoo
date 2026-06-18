"""End-to-end keepalive / timeout / recovery tests through a flaky proxy.

These run real ``OdooConnection`` (no mocks) against the live Odoo through
``tests.helpers.flaky_proxy`` to verify the issue #68 fix at the level the
user actually experiences.

## What's covered here vs at the transport layer

The local dev Odoo (Werkzeug + ``--dev=all``) sends ``Connection: close`` on
every XML-RPC response, so the cached ``HTTPSConnection.sock`` is ``None``
between calls. To reproduce the half-open keepalive scenario from #68 we
front the dev Odoo with an nginx sidecar that terminates client-side
keepalive (see ``tests/docker/test-lb/``):

    Client → FlakyProxy → nginx (LB) → Odoo

With nginx in the chain, the client holds a real keepalive socket. When
``FlakyProxy.silently_drop_upstream()`` fires, the FlakyProxy↔nginx side
goes away invisibly to the client — exactly the half-open state the issue
describes. Tests that need this are gated on the LB being reachable and
auto-skip otherwise.

The remaining tests run directly against the dev Odoo (no LB needed) and
cover regression risks of the timeout itself.

Run the full set with:
    docker compose -f tests/docker/test-lb/compose.yml up -d
    uv run pytest tests/test_keepalive_recovery_e2e.py -v

Marked ``yolo`` so they auto-skip when no Odoo is up.
"""

from __future__ import annotations

import os
import socket
import time
from urllib.parse import urlparse

import pytest

from mcp_server_odoo.config import OdooConfig
from mcp_server_odoo.odoo_connection import OdooConnection, OdooConnectionError
from tests.helpers.flaky_proxy import FlakyProxy

pytestmark = [pytest.mark.yolo]


# ---------------------------------------------------------------------------
# LB sidecar detection — gates the half-open keepalive test
# ---------------------------------------------------------------------------

LB_URL = os.getenv("ODOO_TEST_LB_URL", "http://localhost:8090")


def _lb_reachable(url: str) -> bool:
    """Quick TCP probe; returns True if something is listening on the LB host."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


LB_SKIP_REASON = (
    f"LB sidecar not reachable at {LB_URL}. "
    "Start it with: docker compose -f tests/docker/test-lb/compose.yml up -d"
)


@pytest.fixture
def upstream_host_port() -> tuple[str, int]:
    parsed = urlparse(os.getenv("ODOO_URL", "http://localhost:8069"))
    return parsed.hostname or "localhost", parsed.port or 8069


@pytest.fixture
def proxy(upstream_host_port):
    host, port = upstream_host_port
    p = FlakyProxy(host, port)
    p.start()
    try:
        yield p
    finally:
        p.stop()


def _config_through_proxy(proxy: FlakyProxy) -> OdooConfig:
    """YOLO config that points OdooConnection at our proxy."""
    return OdooConfig(
        url=f"http://127.0.0.1:{proxy.port}",
        database=os.getenv("ODOO_DB"),
        username=os.getenv("ODOO_USER", "admin"),
        password=os.getenv("ODOO_PASSWORD", "admin"),
        yolo_mode="true",
    )


def _connect(config: OdooConfig, timeout: int) -> OdooConnection:
    conn = OdooConnection(config, timeout=timeout)
    conn.connect()
    if not conn.is_authenticated:
        conn.authenticate()
    return conn


def _ping(conn: OdooConnection) -> int:
    """Cheap, side-effect-free call that exercises /xmlrpc/2/object."""
    return conn.search_count("res.partner", [])


# ---------------------------------------------------------------------------
# Baseline — proxy is transparent on the happy path
# ---------------------------------------------------------------------------


def test_proxy_is_transparent_on_happy_path(proxy):
    """Sanity check: with no faults injected, OdooConnection through the proxy
    behaves identically to a direct connection. Catches proxy bugs that would
    otherwise mask test results downstream.
    """
    config = _config_through_proxy(proxy)
    conn = _connect(config, timeout=10)

    results = [_ping(conn) for _ in range(5)]
    assert len(set(results)) == 1, f"five identical calls returned different counts: {results}"
    assert results[0] >= 0


# ---------------------------------------------------------------------------
# Recovery from upstream disconnect (different mechanism than half-open, but
# valuable regression coverage: ``OdooConnection`` reopens correctly)
# ---------------------------------------------------------------------------


def test_pool_recovers_after_upstream_disconnect(proxy):
    """After the proxy drops every active upstream socket, the next tool call
    must succeed within bounded time — exercises the per-call reconnect path
    that ``OdooConnection``/``ConnectionPool`` rely on.

    Note: against local dev Odoo this exits via the "sock is None →
    auto-reconnect" path inside ``http.client.HTTPConnection``, NOT via the
    ``_retry_once_request`` patch (the cached socket is already closed because
    Werkzeug sends ``Connection: close``). The half-open path is covered at
    the transport layer — see this module's docstring.
    """
    socket_timeout = 5
    config = _config_through_proxy(proxy)
    conn = _connect(config, timeout=socket_timeout)

    baseline = _ping(conn)
    assert baseline >= 0

    proxy.silently_drop_upstream()  # close any upstream sockets currently open

    t0 = time.monotonic()
    result = _ping(conn)
    elapsed = time.monotonic() - t0

    assert result == baseline
    assert elapsed < socket_timeout, (
        f"call took {elapsed:.2f}s, expected <{socket_timeout}s "
        f"(reconnect should be near-instant against localhost)"
    )


# ---------------------------------------------------------------------------
# Slow Odoo responses — regression guard for the new 30s socket timeout default
# ---------------------------------------------------------------------------


def test_slow_response_within_timeout_succeeds(proxy):
    """A response that takes a non-trivial fraction of socket_timeout must
    still succeed — the new timeout default must not be triggered prematurely.
    Regression guard against future timeout-tightening that would silently
    break legitimate slow operations (large search_reads, expensive computes).
    """
    delay = 1.5
    socket_timeout = 6
    config = _config_through_proxy(proxy)
    conn = _connect(config, timeout=socket_timeout)
    # Set delay AFTER connect — connect creates its own tunnels through the
    # proxy at delay=0, the next tool call will create a new tunnel that
    # picks up our delay (local Odoo closes connections per request, so each
    # call gets a fresh tunnel).
    proxy.set_response_delay(delay)

    t0 = time.monotonic()
    result = _ping(conn)
    elapsed = time.monotonic() - t0

    assert result >= 0
    assert delay <= elapsed < socket_timeout, (
        f"call took {elapsed:.2f}s, expected ≥{delay}s and <{socket_timeout}s"
    )


# ---------------------------------------------------------------------------
# Half-open keepalive recovery — the actual issue #68 scenario
# Requires the LB sidecar (see tests/docker/test-lb/) to terminate client keepalive.
# ---------------------------------------------------------------------------


@pytest.fixture
def lb_proxy():
    """FlakyProxy in front of the keepalive-terminating LB sidecar.

    The reachability probe lives here (not at module level) so that
    collection never touches the network — unit-only runs import this
    module too.
    """
    if not _lb_reachable(LB_URL):
        pytest.skip(LB_SKIP_REASON)
    parsed = urlparse(LB_URL)
    p = FlakyProxy(parsed.hostname or "127.0.0.1", parsed.port or 80)
    p.start()
    try:
        yield p
    finally:
        p.stop()


def _config_through_lb_proxy(proxy: FlakyProxy) -> OdooConfig:
    return OdooConfig(
        url=f"http://127.0.0.1:{proxy.port}",
        database=os.getenv("ODOO_DB"),
        username=os.getenv("ODOO_USER", "admin"),
        password=os.getenv("ODOO_PASSWORD", "admin"),
        yolo_mode="true",
    )


def test_half_open_keepalive_recovers_via_retry_path(lb_proxy):
    """The actual issue #68 scenario, end-to-end at the OdooConnection level.

    With nginx terminating client-side keepalive, the cached HTTPSConnection
    holds a live socket between calls. When FlakyProxy silently drops the
    upstream side of that connection, the client sees a textbook half-open
    keepalive: writes succeed at TCP level (proxy receives + discards), reads
    block until the socket timeout fires.

    This is the exact code path the half-open retry fix targets:
        write → read times out → ``_retry_once_request`` catches TimeoutError
        → ``self.close()`` → fresh socket → retry → success.

    Without the fix, the read would block until OS-level TCP
    detection or forever — that's the user's reported 4+ minute hang.
    """
    socket_timeout = 3
    config = _config_through_lb_proxy(lb_proxy)
    conn = _connect(config, timeout=socket_timeout)

    # Warm path: prime the cached HTTPSConnection. Verify it actually holds
    # an open keepalive socket (not None) — otherwise we're testing the
    # reconnect-on-every-call path again, not the half-open path.
    baseline = _ping(conn)
    transport = conn.object_proxy._ServerProxy__transport
    cached_host, cached_conn = transport._connection
    assert cached_conn is not None and cached_conn.sock is not None, (
        "expected nginx LB to maintain client-side keepalive — got sock=None. "
        "Check that the LB sidecar is sending Connection: keep-alive."
    )

    # Half-open: close the FlakyProxy↔nginx side without notifying the client.
    # The client's cached socket (client↔FlakyProxy) is now dead but appears
    # alive at userspace level.
    lb_proxy.silently_drop_upstream()

    t0 = time.monotonic()
    result = _ping(conn)
    elapsed = time.monotonic() - t0

    assert result == baseline, "retry result should match warm-path result"

    # Critical assertion #1: the call did NOT hang. Bounded by timeout + retry.
    assert elapsed < socket_timeout + 3, (
        f"call took {elapsed:.2f}s, expected <{socket_timeout + 3}s "
        f"(socket_timeout={socket_timeout}s + retry slack)"
    )
    # Critical assertion #2: the timeout DID fire — the call took at least
    # ~socket_timeout because the first attempt waited for the dead socket
    # to time out before retrying. If this is much faster, we're not actually
    # exercising the half-open path (cached socket was already None).
    assert elapsed >= socket_timeout * 0.8, (
        f"call took only {elapsed:.2f}s — too fast for the half-open path. "
        "Cached socket may have been closed by something other than the drop."
    )


# ---------------------------------------------------------------------------
# Reproducing the reporter's specific 6s × 2 timing pattern from issue #68
# ---------------------------------------------------------------------------


def test_reproduces_reporters_6s_x_2_symptom(proxy):
    """Reproduce the EXACT timing shape reported in #68:

        | T+0.0 s | Tool invoked from Claude                   |
        | T+6.0 s | XML-RPC arrives at Odoo                    |
        | T+6.X s | Odoo responds                              |
        | T+13 s  | Tool result returned to Claude             |

    Using deterministic latency injection (request_delay + response_delay,
    both 6s) we recreate the symptom locally — and document what the fix
    does NOT do for it: with socket_timeout=30 (the production default) a
    ~12s call never trips the timeout, so the retry mechanism doesn't
    engage and the call simply takes ~12s, same as before the fix. This is
    intentional (the 30s default avoids breaking legitimate slow
    operations), but it means the fix does not speed up the reporter's
    median 13s symptom — a future "tighten the timeout to fix #68 median"
    PR has to confront the regression risk.
    """
    request_delay = 6.0
    response_delay = 6.0
    expected_total = request_delay + response_delay  # ≈ 12s, plus ε for Odoo

    # Generous timeout so the call doesn't time out — we want to observe the
    # natural 13s shape, not the truncation. Matches the 30s production default.
    socket_timeout = 30
    config = _config_through_proxy(proxy)
    conn = _connect(config, timeout=socket_timeout)

    proxy.set_request_delay(request_delay)
    proxy.set_response_delay(response_delay)

    t0 = time.monotonic()
    result = _ping(conn)
    elapsed = time.monotonic() - t0

    # Symptom successfully reproduced — the call took ~6s × 2 of injected
    # delay, plus normal request/response overhead (generous upper slack
    # for slow CI runners; the lower bound is the deterministic part).
    assert result >= 0
    assert expected_total <= elapsed < expected_total + 6, (
        f"call took {elapsed:.2f}s, expected ~{expected_total}s "
        f"(reporter's symptom: 13s for similar shape)"
    )


def test_slow_fresh_connection_timeout_is_not_retried(proxy):
    """A timeout on a FRESHLY-opened connection means the server is slow,
    not half-open — the retry guard must NOT engage. The call fails after
    a single attempt at ~1× timeout.

    This is deliberate: retrying a slow request would double the wait and,
    for writes, could double-submit an operation the server is still
    processing. (The half-open retry only fires on REUSED keepalive
    sockets — see ``_retry_once_request``.) Local dev Odoo sends
    ``Connection: close`` per response, so every call here opens fresh.
    """
    socket_timeout = 4
    config = _config_through_proxy(proxy)
    conn = _connect(config, timeout=socket_timeout)
    proxy.set_request_delay(6.0)
    proxy.set_response_delay(6.0)

    t0 = time.monotonic()
    with pytest.raises((OdooConnectionError, TimeoutError, OSError)):
        _ping(conn)
    elapsed = time.monotonic() - t0

    # ONE attempt at ~4s — not two (the unguarded retry would give ~8s)
    assert 3.5 <= elapsed < 6.5, f"got {elapsed:.2f}s, expected ~{socket_timeout}s (single attempt)"


def test_slow_response_exceeding_timeout_fails_bounded(proxy):
    """Operations that legitimately exceed socket_timeout must fail within
    bounded time — they raise an error, but they don't hang. This exercises
    the visible cost of the 30s default timeout: requests that take longer
    than the configured timeout (and whose retry also exceeds it) surface
    as ``OdooConnectionError`` after roughly ``socket_timeout``, not
    indefinitely.
    """
    delay = 4.0
    socket_timeout = 1
    config = _config_through_proxy(proxy)
    # Connect with the short timeout — connect's calls against local Odoo
    # complete in milliseconds, so 1s is plenty. Importantly, no delay is
    # configured on the proxy yet, so connect's tunnels see the upstream at
    # full speed and succeed.
    conn = _connect(config, timeout=socket_timeout)

    # NOW configure the delay. Local Odoo closes the connection per request
    # (Connection: close), so the next tool call will open a fresh tunnel
    # through the proxy that picks up this delay.
    proxy.set_response_delay(delay)

    t0 = time.monotonic()
    with pytest.raises((OdooConnectionError, TimeoutError, OSError)):
        _ping(conn)
    elapsed = time.monotonic() - t0

    # Single attempt (fresh connection — the half-open retry guard does
    # not engage); generous upper bound.
    assert elapsed < socket_timeout * 2 + 3, (
        f"call took {elapsed:.2f}s, expected <{socket_timeout * 2 + 3}s"
    )
