"""Tests for error message sanitization."""

from mcp_server_odoo.error_sanitizer import ErrorSanitizer


class TestErrorSanitizer:
    """Test error message sanitization functionality."""

    def test_sanitize_file_paths(self):
        """Test that file paths are removed."""
        message = 'File "/home/user/odoo/models.py", line 123, in execute'
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "/home/user" not in sanitized
        assert "line 123" not in sanitized
        assert ".py" not in sanitized

    def test_sanitize_module_paths(self):
        """Test that module paths are removed."""
        message = "mcp_server_odoo.odoo_connection: Connection failed"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "mcp_server_odoo." not in sanitized
        assert "Connection failed" in sanitized

    def test_sanitize_class_names(self):
        """Test that class names are removed."""
        message = "Error: <class 'xmlrpc.client.Fault'> occurred"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "<class" not in sanitized
        assert "xmlrpc.client" not in sanitized

    def test_sanitize_memory_addresses(self):
        """Test that memory addresses are removed."""
        message = "Object at 0x7f8b8c0d5f40 not found"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "0x7f8b8c0d5f40" not in sanitized
        assert "Object at" not in sanitized

    def test_sanitize_traceback(self):
        """Test that traceback information is removed."""
        message = """Traceback (most recent call last):
          File "test.py", line 10, in <module>
            raise ValueError("Test error")
        ValueError: Test error"""
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "Traceback" not in sanitized
        assert 'File "test.py"' not in sanitized
        assert "Test error" in sanitized

    def test_field_error_mapping(self):
        """Test specific field error mappings."""
        message = "Invalid field res.partner.invalid_field in leaf ('invalid_field', '=', True)"
        sanitized = ErrorSanitizer.sanitize_message(message)
        # The sanitizer extracts just the field name, not the full model.field path
        assert sanitized == "Invalid field 'invalid_field' in search criteria"

        message = "Field bogus_field does not exist"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized == "Field 'bogus_field' does not exist on this model"

    def test_model_error_mapping(self):
        """Test model error mappings."""
        message = "Model sale.order does not exist"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized == "Model 'sale.order' is not available"

    def test_connection_error_mapping(self):
        """Test connection error mappings."""
        message = "Connection refused"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized == "Cannot connect to Odoo server"

        message = "Operation timeout after 30 seconds"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized == "Request timed out"

    def test_xmlrpc_fault_sanitization(self):
        """Test XML-RPC fault message sanitization."""
        fault = "Access Denied: Invalid API key or insufficient permissions"
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert sanitized == "Access denied: Invalid credentials or insufficient permissions"

        fault = "ValidationError: Field 'vat' is required"
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert sanitized == "Validation error: Please check your input"

        fault = "UserError('Cannot delete record that has dependencies')"
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert sanitized == "Cannot delete record that has dependencies"

    def test_sanitize_missing_error(self):
        """Test that MissingError fault is sanitized to a user-friendly message."""
        fault = "MissingError: Record does not exist or has been deleted."
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert sanitized == "The requested record was not found"

    def test_sanitize_error_details(self):
        """Test error details sanitization."""
        details = {
            "error_type": "ValidationError",
            "traceback": "Full traceback here...",
            "model": "res.partner",
            "operation": "create",
            "internal_path": "/opt/odoo/addons",
        }

        sanitized = ErrorSanitizer.sanitize_error_details(details)

        assert "traceback" not in sanitized
        assert "internal_path" not in sanitized
        assert sanitized["model"] == "res.partner"
        assert sanitized["operation"] == "create"
        assert sanitized["category"] == "validation_error"

    def test_error_type_mapping(self):
        """Test internal error type mapping."""
        assert ErrorSanitizer._map_error_type("ValidationError") == "validation_error"
        assert ErrorSanitizer._map_error_type("OdooConnectionError") == "connection_error"
        assert ErrorSanitizer._map_error_type("NotFoundError") == "not_found"
        assert ErrorSanitizer._map_error_type("UnknownError") == "error"

    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        assert ErrorSanitizer.sanitize_message("") == "An error occurred"
        assert ErrorSanitizer.sanitize_message(None) == "An error occurred"

    def test_preserve_useful_information(self):
        """Test that useful information is preserved."""
        message = "Cannot find partner with email test@example.com"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "test@example.com" in sanitized

        message = "Invalid value 'abc' for integer field"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "'abc'" in sanitized
        assert "integer" in sanitized

    def test_capitalization(self):
        """Test that messages are properly capitalized."""
        message = "connection failed"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized[0].isupper()

    def test_internal_details_removal(self):
        """Test removal of internal implementation details."""
        message = "MCPObjectController: Invalid field res.partner.test_field"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "MCPObjectController:" not in sanitized
        assert "Invalid field" in sanitized

    def test_complex_error_message(self):
        """Test sanitization of complex real-world error."""
        message = """Error executing tool search_records: Connection error: Failed to execute search_count on res.partner: Internal Server Error in MCPObjectController: Invalid field res.partner.invalid_field in leaf ('invalid_field', '=', True)
        File "/opt/odoo/addons/mcp_server/controllers/xmlrpc.py", line 123"""

        sanitized = ErrorSanitizer.sanitize_message(message)

        # Should not contain internal details
        assert "MCPObjectController" not in sanitized
        assert "/opt/odoo" not in sanitized
        assert "line 123" not in sanitized
        assert "search_count" not in sanitized

        # Should contain useful information
        assert "Invalid field" in sanitized


class TestTracebackReduction:
    """Traceback-shaped messages must never leak server internals."""

    UNIQUE_INDEX_FAULT = (
        "Traceback (most recent call last):\n"
        '  File "/opt/odoo/odoo/service/model.py", line 134, in retrying\n'
        "    result = func()\n"
        '  File "/opt/odoo/odoo/models.py", line 4567, in write\n'
        "    self._write(vals)\n"
        "psycopg2.errors.UniqueViolation: duplicate key value violates "
        'unique constraint "res_partner_email_uniq"\n'
        "DETAIL:  Key (email)=(secret@internal.corp) already exists.\n"
    )

    def test_unique_violation_leaks_nothing(self):
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(self.UNIQUE_INDEX_FAULT)
        assert "res_partner_email_uniq" not in sanitized
        assert "secret@internal.corp" not in sanitized
        assert "/opt/odoo" not in sanitized
        assert "_write" not in sanitized
        assert sanitized == "A record with these values already exists"

    def test_value_error_traceback_keeps_final_message_only(self):
        fault = (
            "Traceback (most recent call last):\n"
            '  File "/opt/odoo/odoo/api.py", line 525, in _call_kw\n'
            "    result = getattr(recs, method)(*args, **kwargs)\n"
            "ValueError: Wrong value for res.partner.type: 'bogus'\n"
        )
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert "/opt/odoo" not in sanitized
        assert "_call_kw" not in sanitized
        assert "Wrong value" in sanitized

    def test_modern_user_error_message_preserved(self):
        fault = (
            "Traceback (most recent call last):\n"
            '  File "/opt/odoo/odoo/models.py", line 99, in check\n'
            "    raise UserError(msg)\n"
            "odoo.exceptions.UserError: You cannot delete a posted invoice.\n"
        )
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert sanitized == "You cannot delete a posted invoice."

    def test_multiline_user_error_message_kept_intact(self):
        """Business errors are often multi-line — reduction must keep the
        whole final message, not just the last physical line."""
        fault = (
            "Traceback (most recent call last):\n"
            '  File "/opt/odoo/odoo/models.py", line 99, in check\n'
            "    raise UserError(msg)\n"
            "odoo.exceptions.UserError: You cannot delete this invoice because:\n"
            "- it is posted\n"
            "- it has a payment attached\n"
        )
        sanitized = ErrorSanitizer.sanitize_xmlrpc_fault(fault)
        assert "You cannot delete this invoice because:" in sanitized
        assert "- it is posted" in sanitized
        assert "- it has a payment attached" in sanitized
        assert "/opt/odoo" not in sanitized
        assert "raise UserError" not in sanitized

    def test_missing_error_real_wording_no_placeholder(self):
        # Odoo's actual MissingError wording carries no 'ID <n>' token
        message = "Record res.partner(99,) does not exist or has been deleted"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert "{}" not in sanitized

    def test_failed_to_execute_detail_preserved(self):
        message = "Failed to execute search on res.partner: timeout"
        sanitized = ErrorSanitizer.sanitize_message(message)
        assert sanitized == "Operation failed: timeout"
        assert "{}" not in sanitized

    def test_no_mapping_ever_emits_placeholder(self):
        """Property: no sanitizer output contains a literal '{}'."""
        probes = [
            "Record res.partner(7,) does not exist or has been deleted",
            "Failed to execute write on crm.lead: boom",
            "Invalid field in leaf",
            "Unknown field in domain",
            "Model does not exist",
            "Record does not exist",
        ]
        for probe in probes:
            assert "{}" not in ErrorSanitizer.sanitize_message(probe), probe
