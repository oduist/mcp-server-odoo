"""MCP resource handlers for Odoo data access.

This module implements MCP resources for accessing Odoo data through
standardized URIs using FastMCP decorators.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import Annotations

from .access_control import (
    AccessControlError,
    AccessController,
    AccessControlUnavailableError,
)
from .config import OdooConfig
from .error_handling import (
    ErrorContext,
    MCPPermissionError,
    NotFoundError,
    ValidationError,
)
from .formatters import DatasetFormatter, RecordFormatter
from .logging_config import get_logger, perf_logger
from .odoo_connection import OdooConnection, OdooConnectionError

logger = get_logger(__name__)


class OdooResourceHandler:
    """Handles MCP resource requests for Odoo data."""

    def __init__(
        self,
        app: FastMCP,
        connection: OdooConnection,
        access_controller: AccessController,
        config: OdooConfig,
    ):
        """Initialize resource handler.

        Args:
            app: FastMCP application instance
            connection: Odoo connection instance
            access_controller: Access control instance
            config: Odoo configuration instance
        """
        self.app = app
        self.connection = connection
        self.access_controller = access_controller
        self.config = config

        # Register resources
        self._register_resources()

    async def _ctx_info(self, ctx, message: str):
        """Send info to MCP client context if available."""
        if ctx:
            try:
                await ctx.info(message)
            except Exception:
                logger.debug(f"Failed to send ctx info: {message}")

    def _register_resources(self):
        """Register all resource handlers with FastMCP."""
        # Note: FastMCP uses decorators to register resources.
        # The @self.app.resource decorator automatically handles resource registration.
        # Resources with parameters (like {model}) are registered as templates,
        # not concrete resources, so they won't show in list_resources().

        # Add some concrete resources for enabled models
        # These will show up in the resource list
        self._register_concrete_resources()

        # Register record retrieval resource handler
        @self.app.resource(
            "odoo://{model}/record/{record_id}",
            title="Odoo Record",
            description="Retrieve a specific record from an Odoo model by ID",
            annotations=Annotations(audience=["assistant"], priority=0.5),
        )
        async def get_record(model: str, record_id: str, ctx: Optional[Context] = None) -> str:
            """Retrieve a specific record from Odoo.

            Args:
                model: The Odoo model name (e.g., 'res.partner')
                record_id: The record ID to retrieve

            Returns:
                Formatted record data as text
            """
            return await self._handle_record_retrieval(model, record_id, ctx)

        # Register search resource (no parameters due to FastMCP limitations)
        @self.app.resource(
            "odoo://{model}/search",
            title="Odoo Search",
            description="Search records with default settings (first 10 records)",
            annotations=Annotations(audience=["assistant"], priority=0.5),
        )
        async def search_records(model: str, ctx: Optional[Context] = None) -> str:
            """Search records with default settings.

            Returns first 10 records with all fields.
            For more control, use the search_records tool instead.
            """
            await self._ctx_info(ctx, f"Searching {model} (default: first 10 records)...")
            return await self._handle_search(model, None, None, None, None, None)

        # Note: Browse resource removed due to FastMCP query parameter limitations
        # Use get_record multiple times or search_records tool instead

        # Register count resource (no parameters due to FastMCP limitations)
        @self.app.resource(
            "odoo://{model}/count",
            title="Odoo Record Count",
            description="Count all records in an Odoo model",
            annotations=Annotations(audience=["assistant"], priority=0.3),
        )
        async def count_records(model: str, ctx: Optional[Context] = None) -> str:
            """Count all records in the model.

            For filtered counts, use the search_records tool with limit=0.
            """
            await self._ctx_info(ctx, f"Counting {model} records...")
            return await self._handle_count(model, None)

        # Register fields resource
        @self.app.resource(
            "odoo://{model}/fields",
            title="Odoo Field Definitions",
            description="Get field definitions and metadata for an Odoo model",
            annotations=Annotations(audience=["assistant"], priority=0.4),
        )
        async def get_fields(model: str, ctx: Optional[Context] = None) -> str:
            """Get field definitions for a model.

            Args:
                model: The Odoo model name (e.g., 'res.partner')

            Returns:
                Formatted field definitions and metadata
            """
            await self._ctx_info(ctx, f"Getting field definitions for {model}...")
            return await self._handle_fields(model)

    def _register_concrete_resources(self):
        """Register concrete resources for enabled models.

        Note: In the current FastMCP implementation, resources with parameters
        are registered as templates and won't show in list_resources().
        This is expected behavior - use list_resource_templates() to see them.
        """
        # The template resources registered with decorators are sufficient
        # FastMCP will handle them properly as templates
        pass

    async def _handle_record_retrieval(self, model: str, record_id: str, ctx=None) -> str:
        """Handle record retrieval request.

        Args:
            model: The Odoo model name
            record_id: The record ID to retrieve

        Returns:
            Formatted record data

        Raises:
            NotFoundError: If record doesn't exist
            MCPPermissionError: If access is denied
            ValidationError: For invalid inputs
        """
        context = ErrorContext(model=model, operation="get_record", record_id=record_id)
        await self._ctx_info(ctx, f"Retrieving {model}/{record_id}...")

        logger.info(f"Retrieving record: {model}/{record_id}")

        try:
            with perf_logger.track_operation("resource_get_record", model=model):
                # Validate record ID
                try:
                    record_id_int = int(record_id)
                    if record_id_int <= 0:
                        raise ValueError("Record ID must be positive")
                except ValueError as e:
                    raise ValidationError(
                        f"Invalid record ID '{record_id}': {e}", context=context
                    ) from e

                # Check model access permissions
                try:
                    await asyncio.to_thread(
                        self.access_controller.validate_model_access, model, "read"
                    )
                except AccessControlUnavailableError as e:
                    raise ValidationError(
                        f"Could not verify access (connection error): {e}", context=context
                    ) from e
                except AccessControlError as e:
                    logger.warning(f"Access denied for {model}.read: {e}")
                    raise MCPPermissionError(f"Access denied: {e}", context=context) from e

                # Ensure we're connected
                if not self.connection.is_authenticated:
                    raise ValidationError("Not authenticated with Odoo", context=context)

                # Search for the record to check if it exists
                record_ids = await asyncio.to_thread(
                    self.connection.search, model, [("id", "=", record_id_int)]
                )

                if not record_ids:
                    raise NotFoundError(
                        f"Record not found: {model} with ID {record_id} does not exist",
                        context=context,
                    )

                # Read the record with smart field selection to avoid
                # serialization issues
                safe_fields = await asyncio.to_thread(self._get_safe_fields, model)
                if safe_fields:
                    records = await asyncio.to_thread(
                        self.connection.read, model, record_ids, safe_fields
                    )
                else:
                    # Fallback to all fields if we can't determine safe fields
                    records = await asyncio.to_thread(self.connection.read, model, record_ids)

                if not records:
                    raise NotFoundError(
                        f"Record not found: {model} with ID {record_id} does not exist"
                    )

                record = records[0]

                # Format the record data
                formatted_data = await asyncio.to_thread(self._format_record, model, record)

                logger.info(f"Successfully retrieved record: {model}/{record_id}")
                return formatted_data

        except (NotFoundError, MCPPermissionError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except OdooConnectionError as e:
            logger.error(f"Connection error retrieving {model}/{record_id}: {e}")
            raise ValidationError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving {model}/{record_id}: {e}")
            raise ValidationError(f"Failed to retrieve record: {e}") from e

    async def _handle_search(
        self,
        model: str,
        domain: Optional[str],
        fields: Optional[str],
        limit: Optional[int],
        offset: Optional[int],
        order: Optional[str],
    ) -> str:
        """Handle search request with domain filtering.

        Args:
            model: The Odoo model name
            domain: URL-encoded domain filter
            fields: Comma-separated list of fields
            limit: Maximum records to return
            offset: Pagination offset
            order: Sort order

        Returns:
            Formatted search results with pagination

        Raises:
            MCPPermissionError: If access is denied
            ValidationError: For other errors
        """
        logger.info(f"Searching {model} with domain={domain}, limit={limit}, offset={offset}")

        try:
            # Check model access permissions
            try:
                await asyncio.to_thread(self.access_controller.validate_model_access, model, "read")
            except AccessControlUnavailableError as e:
                raise ValidationError(f"Could not verify access (connection error): {e}") from e
            except AccessControlError as e:
                logger.warning(f"Access denied for {model}.read: {e}")
                raise MCPPermissionError(f"Access denied: {e}") from e

            # Ensure we're connected
            if not self.connection.is_authenticated:
                raise ValidationError("Not authenticated with Odoo")

            # Parse parameters
            parsed_domain = self._parse_domain(domain)
            fields_list = self._parse_fields(fields)
            limit_value = self._parse_limit(limit)
            offset_value = self._parse_offset(offset)
            order_value = self._parse_order(order)

            # Get total count for pagination
            total_count = await asyncio.to_thread(
                self.connection.search_count, model, parsed_domain
            )

            # Perform search
            record_ids = await asyncio.to_thread(
                self.connection.search,
                model,
                parsed_domain,
                limit=limit_value,
                offset=offset_value,
                order=order_value,
            )

            # Read records if any found. Without an explicit field list,
            # restrict to safe fields — reading ALL fields pulls binary/html
            # payloads for every record (the single-record path has the same
            # protection).
            records = []
            if record_ids:
                fields_to_read = fields_list
                if fields_to_read is None:
                    fields_to_read = await asyncio.to_thread(self._get_safe_fields, model)
                records = await asyncio.to_thread(
                    self.connection.read, model, record_ids, fields_to_read
                )

            # Get field metadata for formatting
            try:
                fields_metadata = await asyncio.to_thread(self.connection.fields_get, model)
            except Exception as e:
                logger.debug(f"Could not retrieve field metadata: {e}")
                fields_metadata = None

            # Format search results
            formatted_results = self._format_search_results(
                model,
                records,
                parsed_domain,
                fields_list,
                limit_value,
                offset_value,
                total_count,
                fields_metadata,
            )

            logger.info(f"Search completed: found {len(records)} of {total_count} records")
            return formatted_results

        except (MCPPermissionError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except OdooConnectionError as e:
            logger.error(f"Connection error searching {model}: {e}")
            raise ValidationError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error searching {model}: {e}")
            raise ValidationError(f"Failed to search records: {e}") from e

    def _parse_domain(self, domain: Optional[str]) -> List[Any]:
        """Parse domain parameter from URL-encoded string.

        Args:
            domain: URL-encoded domain string

        Returns:
            Parsed domain list
        """
        if not domain:
            return []

        try:
            # URL decode
            decoded = unquote(domain)
            # Parse JSON
            parsed = json.loads(decoded)

            if not isinstance(parsed, list):
                raise ValueError("Domain must be a list")

            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid domain parameter: {domain} - {e}")
            return []

    def _get_safe_fields(self, model: str) -> Optional[List[str]]:
        """Field names safe to read over XML-RPC.

        Excludes binary/serialized/html fields (which commonly cause
        serialization issues and huge payloads) and private fields.
        Returns None when metadata is unavailable or no safe field exists —
        callers fall back to reading all fields.
        """
        problematic_types = ("binary", "serialized", "html")
        try:
            fields_info = self.connection.fields_get(model)
            safe_fields = [
                field_name
                for field_name, field_info in fields_info.items()
                if field_info.get("type", "") not in problematic_types
                and not field_name.startswith("_")
            ]
        except Exception as e:
            logger.debug(f"Could not get field metadata for {model}: {e}")
            return None
        return safe_fields or None

    def _parse_fields(self, fields: Optional[str]) -> Optional[List[str]]:
        """Parse fields parameter from comma-separated string.

        Args:
            fields: Comma-separated field names

        Returns:
            List of field names or None
        """
        if not fields:
            return None

        # Split and clean field names
        field_list = [f.strip() for f in fields.split(",") if f.strip()]
        return field_list if field_list else None

    def _parse_limit(self, limit: Optional[int]) -> int:
        """Parse and validate limit parameter.

        Args:
            limit: Limit value from request

        Returns:
            Valid limit value
        """
        if limit is None:
            return self.config.default_limit

        # Ensure it's within bounds
        if limit <= 0:
            return self.config.default_limit
        elif limit > self.config.max_limit:
            return self.config.max_limit
        else:
            return limit

    def _parse_offset(self, offset: Optional[int]) -> int:
        """Parse and validate offset parameter.

        Args:
            offset: Offset value from request

        Returns:
            Valid offset value
        """
        if offset is None or offset < 0:
            return 0
        return offset

    def _parse_order(self, order: Optional[str]) -> Optional[str]:
        """Parse and validate order parameter.

        Args:
            order: Order string (e.g., "name asc, id desc")

        Returns:
            Validated order string or None
        """
        if not order:
            return None

        # Basic validation - just ensure it's not empty after stripping
        cleaned = order.strip()
        return cleaned if cleaned else None

    def _format_search_results(
        self,
        model: str,
        records: List[Dict[str, Any]],
        domain: List[Any],
        fields: Optional[List[str]],
        limit: int,
        offset: int,
        total_count: int,
        fields_metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Format search results with pagination metadata.

        Args:
            model: Model name
            records: List of record data
            domain: Applied domain filter
            fields: Requested fields
            limit: Records per page
            offset: Current offset
            total_count: Total matching records
            fields_metadata: Field metadata for formatting

        Returns:
            Formatted search results
        """
        # Calculate pagination info
        current_page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
        has_next = offset + limit < total_count
        has_prev = offset > 0

        # Build pagination hints. Resource URIs cannot carry query
        # parameters (FastMCP routes only the bare odoo://{model}/search
        # template), so point clients at the search_records tool instead
        # of emitting unroutable URIs.
        next_hint = None
        prev_hint = None
        domain_str = json.dumps(domain) if domain else None

        def _tool_hint(page_offset: int) -> str:
            hint = f"use the search_records tool with offset={page_offset}, limit={limit}"
            if domain_str:
                hint += f", domain={domain_str}"
            return hint

        if has_next:
            next_hint = _tool_hint(offset + limit)
        if has_prev:
            prev_hint = _tool_hint(max(0, offset - limit))

        # Use DatasetFormatter for rich formatting
        formatter = DatasetFormatter(model)
        return formatter.format_search_results(
            records=records,
            total_count=total_count,
            limit=limit,
            offset=offset,
            domain=domain,
            fields=fields,
            fields_metadata=fields_metadata,
            next_hint=next_hint,
            prev_hint=prev_hint,
            current_page=current_page,
            total_pages=total_pages,
        )

    async def _handle_count(self, model: str, domain: Optional[str]) -> str:
        """Handle count request with domain filtering.

        Args:
            model: The Odoo model name
            domain: URL-encoded domain filter

        Returns:
            Formatted count result

        Raises:
            MCPPermissionError: If access is denied
            ValidationError: For other errors
        """
        logger.info(f"Counting {model} records with domain: {domain}")

        try:
            # Check model access permissions
            try:
                await asyncio.to_thread(self.access_controller.validate_model_access, model, "read")
            except AccessControlUnavailableError as e:
                raise ValidationError(f"Could not verify access (connection error): {e}") from e
            except AccessControlError as e:
                logger.warning(f"Access denied for {model}.read: {e}")
                raise MCPPermissionError(f"Access denied: {e}") from e

            # Ensure we're connected
            if not self.connection.is_authenticated:
                raise ValidationError("Not authenticated with Odoo")

            # Parse domain
            parsed_domain = self._parse_domain(domain)

            # Get count
            count = await asyncio.to_thread(self.connection.search_count, model, parsed_domain)

            # Format result
            formatted_result = self._format_count_result(model, count, parsed_domain)

            logger.info(f"Count completed: {count} records match criteria")
            return formatted_result

        except (MCPPermissionError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except OdooConnectionError as e:
            logger.error(f"Connection error counting {model}: {e}")
            raise ValidationError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error counting {model}: {e}")
            raise ValidationError(f"Failed to count records: {e}") from e

    async def _handle_fields(self, model: str) -> str:
        """Handle fields request for model introspection.

        Args:
            model: The Odoo model name

        Returns:
            Formatted field definitions

        Raises:
            MCPPermissionError: If access is denied
            ValidationError: For other errors
        """
        logger.info(f"Getting field definitions for {model}")

        try:
            # Check model access permissions
            try:
                await asyncio.to_thread(self.access_controller.validate_model_access, model, "read")
            except AccessControlUnavailableError as e:
                raise ValidationError(f"Could not verify access (connection error): {e}") from e
            except AccessControlError as e:
                logger.warning(f"Access denied for {model}.read: {e}")
                raise MCPPermissionError(f"Access denied: {e}") from e

            # Ensure we're connected
            if not self.connection.is_authenticated:
                raise ValidationError("Not authenticated with Odoo")

            # Get field definitions
            fields = await asyncio.to_thread(self.connection.fields_get, model)

            # Format result
            formatted_result = self._format_fields_result(model, fields)

            logger.info(f"Fields retrieved: {len(fields)} fields found")
            return formatted_result

        except (MCPPermissionError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except OdooConnectionError as e:
            logger.error(f"Connection error getting fields for {model}: {e}")
            raise ValidationError(f"Connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting fields for {model}: {e}")
            raise ValidationError(f"Failed to get field definitions: {e}") from e

    def _format_count_result(self, model: str, count: int, domain: List[Any]) -> str:
        """Format count result.

        Args:
            model: Model name
            count: Record count
            domain: Applied domain filter

        Returns:
            Formatted count result
        """
        lines = [
            f"{'=' * 60}",
            f"Count Result: {model}",
            f"{'=' * 60}",
        ]

        if domain:
            formatter = DatasetFormatter(model)
            lines.append(f"Search criteria: {formatter._format_domain(domain)}")
        else:
            lines.append("Search criteria: All records")

        lines.append("")
        lines.append(f"Total count: {count:,} record(s)")

        return "\n".join(lines)

    def _format_fields_result(self, model: str, fields: Dict[str, Dict[str, Any]]) -> str:
        """Format field definitions result.

        Args:
            model: Model name
            fields: Field definitions dictionary

        Returns:
            Formatted field definitions
        """
        lines = [
            f"{'=' * 60}",
            f"Field Definitions: {model}",
            f"{'=' * 60}",
            f"Total fields: {len(fields)}",
            "",
        ]

        # Group fields by type
        fields_by_type = {}
        for field_name, field_info in sorted(fields.items()):
            field_type = field_info.get("type", "unknown")
            if field_type not in fields_by_type:
                fields_by_type[field_type] = []
            fields_by_type[field_type].append((field_name, field_info))

        # Format fields by type
        for field_type in sorted(fields_by_type.keys()):
            lines.append(f"\n{field_type.upper()} Fields ({len(fields_by_type[field_type])}):")
            lines.append("-" * 30)

            for field_name, field_info in fields_by_type[field_type]:
                lines.append(f"\n{field_name}:")
                lines.append(f"  Label: {field_info.get('string', 'N/A')}")
                lines.append(f"  Required: {field_info.get('required', False)}")
                lines.append(f"  Readonly: {field_info.get('readonly', False)}")

                # Add type-specific information
                if field_type == "selection":
                    selection = field_info.get("selection", [])
                    if selection and len(selection) <= 5:
                        lines.append(
                            f"  Options: {', '.join([f'{k} ({v})' for k, v in selection])}"
                        )
                    elif selection:
                        lines.append(f"  Options: {len(selection)} choices available")

                elif field_type in ("many2one", "one2many", "many2many"):
                    relation = field_info.get("relation", "N/A")
                    lines.append(f"  Related Model: {relation}")

                elif field_type in ("float", "monetary"):
                    digits = field_info.get("digits", "N/A")
                    lines.append(f"  Precision: {digits}")

                # Add help text if available
                help_text = field_info.get("help", "")
                if help_text:
                    lines.append(
                        f"  Help: {help_text[:100]}{'...' if len(help_text) > 100 else ''}"
                    )

        return "\n".join(lines)

    def _format_record(self, model: str, record: Dict[str, Any]) -> str:
        """Format a record for MCP consumption.

        Args:
            model: The model name
            record: The record data

        Returns:
            Formatted text representation
        """
        # Get field metadata if available
        try:
            fields_metadata = self.connection.fields_get(model)
        except Exception as e:
            logger.debug(f"Could not retrieve field metadata: {e}")
            fields_metadata = None

        # Use RecordFormatter for rich formatting
        formatter = RecordFormatter(model)
        return formatter.format_record(record, fields_metadata)


def register_resources(
    app: FastMCP,
    connection: OdooConnection,
    access_controller: AccessController,
    config: OdooConfig,
) -> OdooResourceHandler:
    """Register all Odoo resources with the FastMCP app.

    Args:
        app: FastMCP application instance
        connection: Odoo connection instance
        access_controller: Access control instance
        config: Odoo configuration instance

    Returns:
        The resource handler instance
    """
    handler = OdooResourceHandler(app, connection, access_controller, config)
    logger.info("Registered Odoo MCP resources")
    return handler
