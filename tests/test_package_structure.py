"""Test package structure and basic functionality."""

import subprocess
import sys


class TestPackageStructure:
    """Test the package structure and configuration."""

    def test_package_imports(self):
        """Test that the package can be imported with expected exports."""
        import mcp_server_odoo

        assert hasattr(mcp_server_odoo, "__version__")
        assert hasattr(mcp_server_odoo, "OdooMCPServer")

    def test_cli_help(self):
        """Test CLI help output contains expected content."""
        result = subprocess.run(
            [sys.executable, "-m", "mcp_server_odoo", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        # argparse sends --help to stdout
        assert "Odoo MCP Server" in result.stdout
        assert "ODOO_URL" in result.stdout


class TestDeadCodeStaysDeleted:
    """Guard against zombie re-introduction of deleted machinery via merges.

    These APIs were removed deliberately (2026-06 audit remediation):
    unused record/permission caches, RequestOptimizer, the browse resource
    chain, and the never-wired error conversion/metrics API. If one of
    these names is needed again, reintroduce it consciously with callers
    and tests — don't resurrect it by merge accident.
    """

    def test_performance_module_surface(self):
        import mcp_server_odoo.performance as perf

        for name in ("RequestOptimizer",):
            assert not hasattr(perf, name), f"{name} was deleted; do not reintroduce"
        manager_dead = (
            "get_cached_record",
            "cache_record",
            "invalidate_record_cache",
            "get_cached_permission",
            "cache_permission",
            "optimize_search_fields",
        )
        from mcp_server_odoo.performance import PerformanceManager

        for name in manager_dead:
            assert not hasattr(PerformanceManager, name), f"{name} was deleted"

    def test_error_handling_module_surface(self):
        import mcp_server_odoo.error_handling as eh
        from mcp_server_odoo.error_handling import ErrorHandler, MCPError

        for name in ("handle_odoo_error", "format_user_error", "ErrorMetrics"):
            assert not hasattr(eh, name), f"{name} was deleted"
        for name in ("to_dict", "to_mcp_error"):
            assert not hasattr(MCPError, name), f"MCPError.{name} was deleted"
        for name in ("get_metrics", "get_recent_errors", "clear_metrics", "error_context"):
            assert not hasattr(ErrorHandler, name), f"ErrorHandler.{name} was deleted"

    def test_resources_browse_chain_deleted(self):
        from mcp_server_odoo.resources import OdooResourceHandler

        for name in ("_handle_browse", "_parse_ids", "_format_browse_results"):
            assert not hasattr(OdooResourceHandler, name), f"{name} was deleted"

    def test_uri_schema_browse_operation_deleted(self):
        from mcp_server_odoo.uri_schema import OdooOperation

        assert "BROWSE" not in OdooOperation.__members__


def test_version_single_sourced():
    """__version__, SERVER_VERSION and pyproject.toml must agree.

    v0.5.2 shipped with SERVER_VERSION still at 0.5.1 — this test makes
    that class of release mistake impossible.
    """
    import re
    from pathlib import Path

    import mcp_server_odoo
    from mcp_server_odoo.server import SERVER_VERSION

    # Regex instead of tomllib: the project supports Python 3.10
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert match, "version not found in pyproject.toml"
    pyproject_version = match.group(1)

    assert mcp_server_odoo.__version__ == SERVER_VERSION == pyproject_version
