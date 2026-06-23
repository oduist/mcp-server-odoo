"""Tests for CLI argument / environment / .env precedence in __main__."""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from mcp_server_odoo.config import load_config


class TestEnvFilePrecedence:
    """Values set only in .env must take effect (argparse defaults used to
    be written back to os.environ before load_dotenv ran, masking them)."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        for var in (
            "ODOO_MCP_TRANSPORT",
            "ODOO_MCP_HOST",
            "ODOO_MCP_PORT",
            "ODOO_MCP_LOG_LEVEL",
        ):
            # setenv-then-delenv registers the var with monkeypatch so
            # teardown restores its ORIGINAL state even when code under
            # test (main's env write-back) sets it during the test
            monkeypatch.setenv(var, "sentinel")
            monkeypatch.delenv(var)
        monkeypatch.setenv("ODOO_URL", "http://localhost:8069")
        monkeypatch.setenv("ODOO_API_KEY", "test_key_for_env_file_test")

    def test_dotenv_transport_host_port_take_effect(self, clean_env, tmp_path, monkeypatch):
        for var in ("ODOO_URL", "ODOO_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ODOO_URL=http://envfile-sentinel:8069\n"
            "ODOO_API_KEY=envfile_sentinel_key\n"
            "ODOO_MCP_TRANSPORT=streamable-http\n"
            "ODOO_MCP_HOST=envfile-host\n"
            "ODOO_MCP_PORT=9999\n"
            "ODOO_MCP_LOG_LEVEL=DEBUG\n"
        )

        config = load_config(env_file=env_file)

        # Sentinels prove the FILE was parsed, not the live environment
        assert config.url == "http://envfile-sentinel:8069"
        assert config.api_key == "envfile_sentinel_key"
        assert config.transport == "streamable-http"
        assert config.host == "envfile-host"
        assert config.port == 9999
        assert config.log_level == "DEBUG"

    def test_explicit_cli_flag_overrides_env_file(self, clean_env, tmp_path, monkeypatch):
        """main() writes only explicitly-passed flags back to the environment."""
        from mcp_server_odoo.__main__ import main

        # --version exits before any env write-back; use parse-level check
        # instead: simulate explicit flags and verify env write-back targets
        # only those.
        with patch("mcp_server_odoo.__main__.load_config") as mock_load:
            mock_load.side_effect = ValueError("stop after env write-back")
            main(["--port", "7777"])

        assert os.environ.get("ODOO_MCP_PORT") == "7777"
        assert "ODOO_MCP_TRANSPORT" not in os.environ
        assert "ODOO_MCP_HOST" not in os.environ


class TestRobustStartup:
    def test_non_integer_port_env_gives_friendly_error(self):
        """ODOO_MCP_PORT=abc must not crash --help/--version with a traceback."""
        env = {**os.environ, "ODOO_MCP_PORT": "abc"}
        result = subprocess.run(
            [sys.executable, "-m", "mcp_server_odoo", "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "Traceback" not in result.stderr

    def test_non_integer_slow_threshold_does_not_break_import(self):
        """A bad ODOO_MCP_SLOW_OPERATION_THRESHOLD_MS must not crash imports."""
        env = {**os.environ, "ODOO_MCP_SLOW_OPERATION_THRESHOLD_MS": "abc"}
        result = subprocess.run(
            [sys.executable, "-c", "import mcp_server_odoo"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_slow_threshold_falls_back_to_default(self, monkeypatch):
        from mcp_server_odoo.logging_config import LoggingConfig

        monkeypatch.setenv("ODOO_MCP_SLOW_OPERATION_THRESHOLD_MS", "not-a-number")
        assert LoggingConfig().slow_operation_threshold_ms == 1000

        monkeypatch.setenv("ODOO_MCP_SLOW_OPERATION_THRESHOLD_MS", "250")
        assert LoggingConfig().slow_operation_threshold_ms == 250
