"""Tests for error handling and logging system."""

import json
import logging
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_odoo.error_handling import (
    AuthenticationError,
    ConfigurationError,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    MCPConnectionError,
    MCPError,
    MCPPermissionError,
    MCPSystemError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    error_handler,
)
from mcp_server_odoo.logging_config import (
    LoggingConfig,
    PerformanceLogger,
    RequestLoggingAdapter,
    StructuredFormatter,
    log_request,
    log_response,
    perf_logger,
    setup_logging,
)


class TestMCPError:
    """Test the MCPError base class."""

    def test_error_creation(self):
        """Test creating an MCPError with all parameters."""
        context = ErrorContext(
            model="res.partner",
            operation="search",
            record_id=42,
            user_id=1,
            request_id="test-123",
        )

        error = MCPError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            code="TEST_ERROR",
            details={"field": "email", "value": "invalid"},
            context=context,
        )

        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.code == "TEST_ERROR"
        assert error.details == {"field": "email", "value": "invalid"}
        assert error.context.model == "res.partner"
        assert error.context.operation == "search"

    def test_error_code_generation(self):
        """Test automatic error code generation."""
        error = AuthenticationError("Invalid credentials")
        assert error.code == "AUTH_ERROR"

        error = MCPPermissionError("Access denied")
        assert error.code == "PERMISSION_DENIED"

        error = NotFoundError("Record not found")
        assert error.code == "NOT_FOUND"

        error = ValidationError("Invalid input")
        assert error.code == "VALIDATION_ERROR"

        error = MCPConnectionError("Connection failed")
        assert error.code == "CONNECTION_ERROR"

        error = MCPSystemError("System failure")
        assert error.code == "SYSTEM_ERROR"

        error = ConfigurationError("Bad config")
        assert error.code == "CONFIG_ERROR"

        error = RateLimitError("Too many requests")
        assert error.code == "RATE_LIMIT_EXCEEDED"


class TestErrorHandler:
    """Test the ErrorHandler class."""

    def test_handle_mcp_error(self):
        """Test handling an MCPError."""
        handler = ErrorHandler()

        error = ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            handler.handle_error(error)

    def test_handle_standard_exception(self):
        """Test converting standard exceptions to MCPError."""
        handler = ErrorHandler()

        # Test ValueError conversion
        with pytest.raises(ValidationError) as exc_info:
            handler.handle_error(ValueError("Invalid value"))
        assert "Invalid input: Invalid value" in str(exc_info.value)

        # Test ConnectionRefusedError conversion
        with pytest.raises(MCPConnectionError) as exc_info:
            handler.handle_error(ConnectionRefusedError("Connection refused"))
        assert "Connection failed:" in str(exc_info.value)

        # Test KeyError conversion
        with pytest.raises(NotFoundError) as exc_info:
            handler.handle_error(KeyError("missing_key"))
        assert "Resource not found:" in str(exc_info.value)

        # Test generic exception conversion
        with pytest.raises(MCPSystemError) as exc_info:
            handler.handle_error(RuntimeError("Something went wrong"))
        assert "Unexpected error:" in str(exc_info.value)

    def test_handle_error_no_reraise(self):
        """Test handling error without re-raising."""
        handler = ErrorHandler()
        error = ValidationError("Test error")

        result = handler.handle_error(error, reraise=False)

        assert isinstance(result, MCPError)
        assert result.message == "Test error"


class TestLoggingConfiguration:
    """Test logging configuration and utilities."""

    def test_structured_formatter(self):
        """Test JSON log formatting."""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.error_code = "TEST_ERROR"
        record.model = "res.partner"

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["logger"] == "test.logger"
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["error_code"] == "TEST_ERROR"
        assert log_data["model"] == "res.partner"
        assert "timestamp" in log_data

    def test_request_logging_adapter(self):
        """Test request logging adapter."""
        logger = logging.getLogger("test")
        adapter = RequestLoggingAdapter(logger, request_id="test-123")

        assert adapter.request_id == "test-123"

        # Test that request ID is added to extra
        msg, kwargs = adapter.process("Test message", {})
        assert kwargs["extra"]["request_id"] == "test-123"

    def test_performance_logger(self):
        """Test performance tracking."""
        logger = MagicMock()
        perf = PerformanceLogger(logger)

        with perf.track_operation("test_op", model="res.partner"):
            time.sleep(0.01)  # Small delay

        # Check that info was logged
        logger.info.assert_called()
        call_args = logger.info.call_args
        assert "test_op" in call_args[0][0]
        assert "completed in" in call_args[0][0]
        assert call_args[1]["extra"]["operation"] == "test_op"
        assert call_args[1]["extra"]["model"] == "res.partner"
        assert call_args[1]["extra"]["duration_ms"] > 0

    def test_setup_logging(self):
        """Test logging setup writes valid JSON to file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            setup_logging(
                log_level="DEBUG",
                use_json=True,
                log_file=tmp.name,
            )

            logger = logging.getLogger("test")
            logger.debug("Test debug message")

            # Force flush all handlers
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Read and verify file content
            with open(tmp.name) as f:
                content = f.read()
            assert len(content) > 0
            # Verify at least one line is valid JSON
            for line in content.strip().split("\n"):
                if line.strip():
                    parsed = json.loads(line)
                    assert "message" in parsed

            # Clean up
            os.unlink(tmp.name)

    def test_logging_config_from_env(self):
        """Test loading logging config from environment."""
        with patch.dict(
            os.environ,
            {
                "ODOO_MCP_LOG_LEVEL": "DEBUG",
                "ODOO_MCP_LOG_JSON": "true",
                "ODOO_MCP_LOG_FILE": "/tmp/test.log",
                "ODOO_MCP_SLOW_OPERATION_THRESHOLD_MS": "500",
            },
        ):
            config = LoggingConfig()

            assert config.log_level == "DEBUG"
            assert config.use_json is True
            assert config.log_file == "/tmp/test.log"
            assert config.slow_operation_threshold_ms == 500

    def test_log_request_response(self):
        """Test request/response logging helpers."""
        logger = MagicMock()

        # Test request logging
        log_request(
            logger,
            method="GET",
            path="/api/test",
            params={"limit": 10},
            body={"filter": "active"},
        )

        logger.info.assert_called()
        call_args = logger.info.call_args
        assert "GET /api/test" in call_args[0][0]
        assert call_args[1]["extra"]["request_method"] == "GET"
        assert call_args[1]["extra"]["request_params"] == {"limit": 10}

        # Test response logging
        log_response(
            logger,
            status="200 OK",
            duration_ms=123.45,
            response_size=1024,
        )

        assert logger.info.call_count == 2
        call_args = logger.info.call_args
        assert "200 OK (123.45ms)" in call_args[0][0]
        assert call_args[1]["extra"]["response_status"] == "200 OK"
        assert call_args[1]["extra"]["response_size"] == 1024

        # Test error response logging
        log_response(
            logger,
            status="500 Error",
            duration_ms=50.0,
            error="Internal server error",
        )

        logger.error.assert_called()
        call_args = logger.error.call_args
        assert "500 Error" in call_args[0][0]
        assert "Internal server error" in call_args[0][0]


class TestGlobalInstances:
    """Test global error handler and logging instances."""

    def test_global_error_handler(self):
        """Test that global error handler works correctly."""
        # Generate an error
        with pytest.raises(ValidationError):
            error_handler.handle_error(ValueError("Test"))

    def test_global_perf_logger(self, caplog):
        """Test that global performance logger tracks operations."""
        with caplog.at_level(logging.DEBUG):
            with perf_logger.track_operation("test_operation"):
                time.sleep(0.01)

        # Verify the operation was actually tracked — check real log output
        perf_messages = [r.message for r in caplog.records if "test_operation" in r.message]
        assert len(perf_messages) >= 1, "Performance logger should have logged the operation"
        assert "completed in" in perf_messages[0]


class TestBuiltinExceptionClassification:
    """Builtin OS-level exceptions must classify correctly (the custom
    classes used to shadow the builtins, making these branches dead)."""

    def test_builtin_permission_error_classified_as_permission(self):
        handler = ErrorHandler()
        result = handler.handle_error(PermissionError("disk says no"), reraise=False)
        assert isinstance(result, MCPPermissionError)
        assert result.category == ErrorCategory.PERMISSION
        assert result.severity == ErrorSeverity.MEDIUM

    def test_builtin_connection_errors_classified_as_connection(self):
        handler = ErrorHandler()
        for exc in (
            ConnectionResetError("reset"),
            BrokenPipeError("pipe"),
            ConnectionRefusedError("refused"),
        ):
            result = handler.handle_error(exc, reraise=False)
            assert isinstance(result, MCPConnectionError), type(exc).__name__
            assert result.category == ErrorCategory.CONNECTION

    def test_unknown_exception_still_system_error(self):
        handler = ErrorHandler()
        result = handler.handle_error(RuntimeError("boom"), reraise=False)
        assert isinstance(result, MCPSystemError)
