"""Pytest configuration and fixtures for Odoo MCP Server tests."""

import functools
import os
import socket
import xmlrpc.client

import pytest
from dotenv import load_dotenv

from mcp_server_odoo.config import OdooConfig

# Load .env file for tests
load_dotenv()

# Import model discovery helper
try:
    from tests.helpers.model_discovery import ModelDiscovery

    MODEL_DISCOVERY_AVAILABLE = True
except ImportError:
    MODEL_DISCOVERY_AVAILABLE = False


def is_odoo_server_available(host: str = "localhost", port: int = 8069) -> bool:
    """Check if Odoo server is available at the given host and port."""
    try:
        # Try to connect to the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            return False

        # Try to access the XML-RPC endpoint
        try:
            proxy = xmlrpc.client.ServerProxy(f"http://{host}:{port}/xmlrpc/2/common")
            proxy.version()
            return True
        except Exception:
            return False

    except Exception:
        return False


def _parse_odoo_host_port() -> tuple[str, int]:
    from urllib.parse import urlparse

    url = os.getenv("ODOO_URL", "http://localhost:8069")
    parsed = urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 8069


# Probes are LAZY and memoized: unit-only runs must not touch the network.
# They are triggered from pytest_collection_modifyitems only when yolo/mcp
# tests were actually collected (and survive the -m filter).
@functools.lru_cache(maxsize=1)
def odoo_server_available() -> bool:
    """Whether a (vanilla) Odoo answers XML-RPC at ODOO_URL."""
    host, port = _parse_odoo_host_port()
    return is_odoo_server_available(host, port)


@functools.lru_cache(maxsize=1)
def mcp_module_available() -> bool:
    """Whether the Odoo at ODOO_URL also has the MCP module installed.

    A vanilla Odoo (the YOLO scenario) serves /xmlrpc/2/* but not the
    /mcp/ REST routes — mcp-marked tests must skip there instead of
    failing with confusing 404/auth errors.
    """
    if not odoo_server_available():
        return False
    url = os.getenv("ODOO_URL", "http://localhost:8069").rstrip("/") + "/mcp/health"
    try:
        import urllib.request

        # Multi-DB instances can't route /mcp/health without an explicit database
        request = urllib.request.Request(url)
        db = os.getenv("ODOO_DB")
        if db:
            request.add_header("X-Odoo-Database", db)
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "yolo: needs running Odoo instance (vanilla XML-RPC)")
    config.addinivalue_line("markers", "mcp: needs running Odoo with MCP module installed")


def pytest_collection_modifyitems(config, items):
    """Skip yolo/mcp tests when their backing services are unavailable."""
    # Respect the -m filter: only probe for tests that will actually run
    selected = [item for item in items if not _deselected_by_mark_filter(config, item)]
    needs_server = any("yolo" in item.keywords or "mcp" in item.keywords for item in selected)
    if not needs_server:
        return

    host, port = _parse_odoo_host_port()
    if not odoo_server_available():
        skip_odoo = pytest.mark.skip(reason=f"Odoo server not available at {host}:{port}")
        for item in items:
            if "yolo" in item.keywords or "mcp" in item.keywords:
                item.add_marker(skip_odoo)
        return

    needs_mcp = any("mcp" in item.keywords for item in selected)
    if needs_mcp and not mcp_module_available():
        skip_mcp = pytest.mark.skip(
            reason="Odoo MCP module not available (GET /mcp/health failed) — vanilla Odoo?"
        )
        for item in items:
            if "mcp" in item.keywords:
                item.add_marker(skip_mcp)


def _deselected_by_mark_filter(config, item) -> bool:
    """Whether the -m expression deselects this item."""
    markexpr = config.option.markexpr
    if not markexpr:
        return False
    try:
        # Private pytest API (import and signatures have churned across
        # pytest releases) — if it changes shape, fall back to "selected":
        # worst case an unnecessary server probe, never a wrong skip
        from _pytest.mark.expression import Expression

        return not Expression.compile(markexpr).evaluate(lambda name: name in item.keywords)
    except Exception:
        return False


@pytest.fixture
def odoo_server_required():
    """Fixture that skips test if Odoo server is not available."""
    if not odoo_server_available():
        host, port = _parse_odoo_host_port()
        pytest.skip(f"Odoo server not available at {host}:{port}")


@pytest.fixture
def test_config_with_server_check(odoo_server_required) -> OdooConfig:
    """Create test configuration, but skip if server not available."""
    # Require environment variables to be set
    if not os.getenv("ODOO_URL"):
        pytest.skip("ODOO_URL environment variable not set. Please configure .env file.")

    if not os.getenv("ODOO_API_KEY") and not os.getenv("ODOO_PASSWORD"):
        pytest.skip("Neither ODOO_API_KEY nor ODOO_PASSWORD set. Please configure .env file.")

    return OdooConfig(
        url=os.getenv("ODOO_URL"),
        api_key=os.getenv("ODOO_API_KEY") or None,
        username=os.getenv("ODOO_USER") or None,
        password=os.getenv("ODOO_PASSWORD") or None,
        database=os.getenv("ODOO_DB"),  # DB can be auto-detected
        log_level=os.getenv("ODOO_MCP_LOG_LEVEL", "INFO"),
        default_limit=int(os.getenv("ODOO_MCP_DEFAULT_LIMIT", "10")),
        max_limit=int(os.getenv("ODOO_MCP_MAX_LIMIT", "100")),
    )


# MCP Model Discovery Fixtures
# These fixtures help make tests model-agnostic by discovering
# and adapting to whatever models are currently available


@pytest.fixture
def model_discovery():
    """Create a model discovery helper.

    Creates a fresh discovery instance for each test.
    """
    if not MODEL_DISCOVERY_AVAILABLE:
        pytest.skip("Model Discovery not available")

    if not odoo_server_available():
        pytest.skip("Odoo server not available")

    # Create config for discovery
    config = OdooConfig(
        url=os.getenv("ODOO_URL"),
        api_key=os.getenv("ODOO_API_KEY") or None,
        username=os.getenv("ODOO_USER") or None,
        password=os.getenv("ODOO_PASSWORD") or None,
        database=os.getenv("ODOO_DB"),
    )

    discovery = ModelDiscovery(config)
    return discovery


@pytest.fixture
def readable_model(model_discovery):
    """Get a model with read permission.

    Skips test if no readable models are available.
    """
    return model_discovery.require_readable_model()


@pytest.fixture
def writable_model(model_discovery):
    """Get a model with write permission.

    Skips test if no writable models are available.
    """
    return model_discovery.require_writable_model()


@pytest.fixture
def disabled_model(model_discovery):
    """Get a model name that is NOT enabled.

    Returns a model that should fail access checks.
    """
    return model_discovery.get_disabled_model()


@pytest.fixture
def test_models(model_discovery):
    """Get commonly available models for testing.

    Returns a list of models that are commonly enabled,
    or skips if none are available.
    """
    models = model_discovery.get_common_models()
    if not models:
        models = [model_discovery.require_readable_model()]
    return models
