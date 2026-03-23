"""
tools/sources/openapi.py — OpenAPIToolSource turns REST APIs into LLM tools.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.

Given an OpenAPI 3.x spec (URL, file path, or dict), this source:
1. Parses the spec at initialize() time
2. Builds LLM tool definitions from operationIds
3. Executes HTTP requests on call()
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ninetrix.tools.base import ToolSource
from ninetrix._internals.types import ToolError, ConfigurationError

_log = logging.getLogger("ninetrix.tools.sources.openapi")


class OpenAPIToolSource(ToolSource):
    """Makes any REST API with an OpenAPI 3.x spec into LLM-callable tools.

    Args:
        spec:         URL, file path, or pre-parsed dict of the OpenAPI spec.
        name_prefix:  String prepended to tool names, e.g. ``"stripe__"``.
        actions:      Optional list of operationId strings to expose.
                      If None/empty, all operations with operationIds are exposed.
        base_url:     Override the base URL from the spec's ``servers[0]``.
        auth:         Raw auth config dict from agentfile.yaml
                      (passed to ``AuthResolver.resolve_headers()``).
        ctx:          :class:`~ninetrix.tools.agent_context.AgentContext` (optional).

    Example::

        source = OpenAPIToolSource(
            spec="https://petstore3.swagger.io/api/v3/openapi.json",
            name_prefix="petstore__",
            actions=["findPetsByStatus", "getPetById"],
            auth={"type": "bearer", "token": "${PET_API_KEY}"},
        )
        await source.initialize()
        result = await source.call("petstore__findPetsByStatus", {"status": "available"})
    """

    source_type = "openapi"

    def __init__(
        self,
        spec: str | dict,
        *,
        name_prefix: str = "",
        actions: list[str] | None = None,
        base_url: str = "",
        auth: dict | None = None,
        ctx: Any | None = None,
    ) -> None:
        self._spec_ref = spec
        self._name_prefix = name_prefix
        self._actions_filter = set(actions) if actions else None
        self._base_url_override = base_url
        self._auth = auth
        self._ctx = ctx
        self._operations: dict[str, _Operation] = {}
        self._base_url = ""

    def validate_config(self) -> None:
        if not self._spec_ref:
            raise ConfigurationError(
                "OpenAPIToolSource requires a spec (URL, path, or dict).\n"
                "  Fix: pass spec='https://...' or spec={'openapi': '3.0', ...}"
            )

    async def initialize(self) -> None:
        """Fetch, parse, and index the OpenAPI spec."""
        spec = await self._load_spec()
        self._base_url = self._base_url_override or self._extract_base_url(spec)
        self._parse_operations(spec)

    def tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": op.summary,
                    "parameters": op.parameters_schema,
                },
            }
            for tool_name, op in self._operations.items()
        ]

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._operations

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Build and execute an HTTP request from the tool call arguments."""
        if tool_name not in self._operations:
            raise ToolError(
                f"OpenAPI tool '{tool_name}' not found.\n"
                "  Fix: check tool name matches an operationId in the spec."
            )

        op = self._operations[tool_name]
        http = self._get_http()

        # Build URL with path parameter interpolation
        url = self._base_url + self._interpolate_path(op.path, arguments)

        # Build headers (auth + content-type)
        headers: dict[str, str] = {}
        if self._ctx and self._ctx.auth:
            headers.update(self._ctx.auth.resolve_headers(self._auth))
        elif self._auth:
            # Fallback: inline auth resolution without AuthResolver
            from ninetrix.tools.auth_resolver import AuthResolver
            headers.update(AuthResolver().resolve_headers(self._auth))

        # Split arguments into query params and body
        query_params: dict[str, Any] = {}
        body: dict[str, Any] | None = None

        for arg_name, arg_value in arguments.items():
            param_loc = op.param_locations.get(arg_name)
            if param_loc == "path":
                continue  # already interpolated
            elif param_loc == "query":
                query_params[arg_name] = arg_value
            elif param_loc == "header":
                headers[arg_name] = str(arg_value)
            else:
                # Unknown location or body field
                if body is None:
                    body = {}
                body[arg_name] = arg_value

        # If the operation has a requestBody, all non-path/query args go to body
        if op.has_request_body and body is None:
            body = {}

        _log.debug(f"openapi call: {op.method.upper()} {url}")

        resp = await http.request(
            method=op.method.upper(),
            url=url,
            headers=headers,
            params=query_params or None,
            json=body,
            timeout=60.0,
        )

        if resp.status_code >= 400:
            return f"HTTP {resp.status_code}: {resp.text[:500]}"

        # Return response body
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            try:
                return json.dumps(resp.json(), indent=2)
            except Exception:
                pass
        return resp.text

    async def health_check(self) -> bool:
        """HEAD request to base URL."""
        if not self._base_url:
            return False
        try:
            http = self._get_http()
            resp = await http.head(self._base_url, timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    # ── Private helpers ───────────────────────────────────────────────

    def _get_http(self) -> Any:
        if self._ctx and self._ctx.http:
            return self._ctx.http
        from ninetrix._internals.http import get_http_client
        return get_http_client()

    async def _load_spec(self) -> dict:
        """Load the OpenAPI spec from URL, file path, or dict."""
        if isinstance(self._spec_ref, dict):
            return self._spec_ref

        spec_str = str(self._spec_ref)

        # URL — fetch via HTTP
        if spec_str.startswith("http://") or spec_str.startswith("https://"):
            http = self._get_http()
            resp = await http.get(spec_str, timeout=30.0)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "yaml" in content_type or spec_str.endswith((".yaml", ".yml")):
                try:
                    import yaml  # type: ignore[import-untyped]
                    return yaml.safe_load(resp.text)
                except ImportError:
                    raise ConfigurationError(
                        "YAML spec requires pyyaml. Fix: pip install pyyaml"
                    )
            return resp.json()

        # File path
        path = Path(spec_str)
        if not path.exists():
            raise ConfigurationError(
                f"OpenAPI spec file not found: {spec_str}\n"
                "  Fix: check the file path."
            )
        text = path.read_text()
        if spec_str.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore[import-untyped]
                return yaml.safe_load(text)
            except ImportError:
                raise ConfigurationError(
                    "YAML spec requires pyyaml. Fix: pip install pyyaml"
                )
        return json.loads(text)

    @staticmethod
    def _extract_base_url(spec: dict) -> str:
        servers = spec.get("servers", [])
        if servers and isinstance(servers, list):
            return servers[0].get("url", "").rstrip("/")
        return ""

    def _parse_operations(self, spec: dict) -> None:
        """Walk spec paths and build _Operation entries."""
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if method.startswith("x-") or not isinstance(operation, dict):
                    continue
                if method not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                    continue

                op_id = operation.get("operationId")
                if not op_id:
                    # Auto-generate: get_users_by_id
                    slug = re.sub(r"[{}]", "", path).replace("/", "_").strip("_")
                    op_id = f"{method}_{slug}"

                # Apply action filter
                if self._actions_filter and op_id not in self._actions_filter:
                    continue

                tool_name = f"{self._name_prefix}{op_id}"

                # Build parameters schema
                params_schema, param_locations = self._build_params_schema(
                    operation.get("parameters", []),
                    operation.get("requestBody"),
                )

                self._operations[tool_name] = _Operation(
                    method=method,
                    path=path,
                    summary=operation.get("summary") or operation.get("description") or op_id,
                    parameters_schema=params_schema,
                    param_locations=param_locations,
                    has_request_body=operation.get("requestBody") is not None,
                )

    @staticmethod
    def _build_params_schema(
        parameters: list[dict],
        request_body: dict | None,
    ) -> tuple[dict, dict[str, str]]:
        """Convert OpenAPI parameters + requestBody to a flat JSON Schema.

        Returns:
            (json_schema, param_locations) where param_locations maps
            param_name -> "path" | "query" | "header".
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        param_locations: dict[str, str] = {}

        # Path/query/header parameters
        for param in parameters:
            if not isinstance(param, dict):
                continue
            name = param.get("name", "")
            if not name:
                continue
            location = param.get("in", "query")
            param_locations[name] = location

            schema = param.get("schema", {"type": "string"})
            prop: dict[str, Any] = {}
            if "type" in schema:
                prop["type"] = schema["type"]
            if "description" in param:
                prop["description"] = param["description"]
            elif "description" in schema:
                prop["description"] = schema["description"]
            if "enum" in schema:
                prop["enum"] = schema["enum"]
            if "default" in schema:
                prop["default"] = schema["default"]
            properties[name] = prop

            if param.get("required", location == "path"):
                required.append(name)

        # Request body — flatten top-level properties
        if request_body and isinstance(request_body, dict):
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            body_schema = json_content.get("schema", {})
            body_props = body_schema.get("properties", {})
            body_required = body_schema.get("required", [])

            for prop_name, prop_schema in body_props.items():
                clean: dict[str, Any] = {}
                if "type" in prop_schema:
                    clean["type"] = prop_schema["type"]
                if "description" in prop_schema:
                    clean["description"] = prop_schema["description"]
                if "enum" in prop_schema:
                    clean["enum"] = prop_schema["enum"]
                properties[prop_name] = clean
                # Don't set param_locations — body fields have no location

            required.extend(body_required)

        return (
            {
                "type": "object",
                "properties": properties,
                **({"required": required} if required else {}),
            },
            param_locations,
        )

    @staticmethod
    def _interpolate_path(path: str, arguments: dict[str, Any]) -> str:
        """Replace {param} placeholders in path with argument values."""
        result = path
        for key, value in arguments.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


class _Operation:
    """Internal representation of a single API operation."""

    __slots__ = ("method", "path", "summary", "parameters_schema", "param_locations", "has_request_body")

    def __init__(
        self,
        method: str,
        path: str,
        summary: str,
        parameters_schema: dict,
        param_locations: dict[str, str],
        has_request_body: bool,
    ) -> None:
        self.method = method
        self.path = path
        self.summary = summary
        self.parameters_schema = parameters_schema
        self.param_locations = param_locations
        self.has_request_body = has_request_body
