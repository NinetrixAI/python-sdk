"""
tools/auth_resolver.py — converts auth config dicts into HTTP headers.

Layer: L2 (tools) — imports stdlib only.

The AuthResolver is a shared utility injected via :class:`AgentContext`.
It takes raw ``auth:`` dicts from agentfile.yaml tool configs and produces
HTTP headers (or query params) at call time.

Stateless: reads env vars on every call so rotated credentials are picked up
without agent restart.

Supported auth types::

    # Bearer token
    auth: {type: bearer, token: "${GITHUB_TOKEN}"}

    # Basic auth
    auth: {type: basic, username: "${USER}", password: "${PASS}"}

    # Custom header
    auth: {type: header, header_name: "X-API-Key", token: "${API_KEY}"}

    # API key as query parameter
    auth: {type: api_key_query, query_param: "api_key", token: "${KEY}"}
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Callable

_log = logging.getLogger("ninetrix.tools.auth_resolver")


class AuthResolver:
    """Converts auth config dicts into HTTP headers at call time.

    Args:
        env: Callable ``(key, default) -> value`` for reading env vars.
             Defaults to ``os.environ.get``.
    """

    def __init__(self, env: Callable[..., str] = os.environ.get) -> None:
        self._env = env

    def _resolve_value(self, raw: str) -> str:
        """Resolve ``${VAR}`` references in a value string."""
        if not raw:
            return ""
        # Handle ${VAR} syntax — resolve from environment
        if raw.startswith("${") and raw.endswith("}"):
            var_name = raw[2:-1]
            return self._env(var_name, "") or ""
        return raw

    def resolve_headers(self, auth: dict | None) -> dict[str, str]:
        """Return HTTP headers for the given auth config dict.

        Args:
            auth: Raw auth dict from agentfile.yaml, e.g.
                  ``{"type": "bearer", "token": "${GITHUB_TOKEN}"}``.
                  If None or empty, returns empty dict.

        Returns:
            Dict of HTTP headers to include in the request.
        """
        if not auth:
            return {}

        auth_type = auth.get("type", "bearer")

        if auth_type == "bearer":
            token = self._resolve_value(auth.get("token", ""))
            if token:
                return {"Authorization": f"Bearer {token}"}
            return {}

        if auth_type == "basic":
            username = self._resolve_value(auth.get("username", ""))
            password = self._resolve_value(auth.get("password", ""))
            if username or password:
                encoded = base64.b64encode(
                    f"{username}:{password}".encode()
                ).decode()
                return {"Authorization": f"Basic {encoded}"}
            return {}

        if auth_type == "header":
            header_name = auth.get("header_name", "")
            token = self._resolve_value(auth.get("token", ""))
            if header_name and token:
                return {header_name: token}
            return {}

        if auth_type == "api_key_query":
            # Query params handled by resolve_query_params() — no headers
            return {}

        _log.warning(f"Unknown auth type: {auth_type!r}")
        return {}

    def resolve_query_params(self, auth: dict | None) -> dict[str, str]:
        """Return query parameters for api_key_query auth type.

        Args:
            auth: Raw auth dict from agentfile.yaml.

        Returns:
            Dict of query parameters to include in the request URL.
        """
        if not auth or auth.get("type") != "api_key_query":
            return {}

        param_name = auth.get("query_param", "api_key")
        token = self._resolve_value(auth.get("token", ""))
        if token:
            return {param_name: token}
        return {}
