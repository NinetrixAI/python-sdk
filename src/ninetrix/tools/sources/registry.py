"""
tools/sources/registry.py — RegistryToolSource for Ninetrix Skill Registry.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.
"""

from __future__ import annotations

import logging
from typing import Any

from ninetrix.tools.base import ToolSource
from ninetrix._internals.types import ToolError
from ninetrix._internals.http import get_http_client
from ninetrix._internals.tenant import get_tenant

_log = logging.getLogger("ninetrix.runtime.dispatcher")


class RegistryToolSource(ToolSource):
    """Loads skills from the Ninetrix Skill Registry on first use.

    Lazy initialisation: schemas are fetched once via HTTP at
    ``initialize()`` time.  Tool calls are proxied to the registry API —
    no local Python code is downloaded.

    Args:
        skills:       List of skill names to load.
        registry_url: Base URL of the Ninetrix Skill Registry.
        api_key:      Bearer token for registry authentication.
        ctx:          Optional :class:`~ninetrix.tools.agent_context.AgentContext`.
    """

    source_type = "registry"

    def __init__(
        self,
        skills: list[str],
        registry_url: str,
        api_key: str,
        ctx: Any | None = None,
    ) -> None:
        self._skills = skills
        self._registry_url = registry_url.rstrip("/")
        self._api_key = api_key
        self._ctx = ctx
        self._schemas: list[dict] = []
        self._skill_names: set[str] = set()

    def _http(self) -> Any:
        """Return the HTTP client — prefer AgentContext, fall back to singleton."""
        if self._ctx is not None and self._ctx.http is not None:
            return self._ctx.http
        return get_http_client()

    async def initialize(self) -> None:
        """Fetch schemas for the requested skills from the registry."""
        tenant = get_tenant()
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if tenant:
            headers["X-Org-ID"] = tenant.org_id

        client = self._http()
        resp = await client.post(
            f"{self._registry_url}/v1/skills/schemas",
            json={"skills": self._skills},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        self._schemas = data.get("schemas", [])
        self._skill_names = {
            s["function"]["name"] for s in self._schemas if "function" in s
        }

    def tool_definitions(self) -> list[dict]:
        return list(self._schemas)

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._skill_names

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Proxy the call to the registry execution endpoint."""
        tenant = get_tenant()
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        if tenant:
            headers["X-Org-ID"] = tenant.org_id

        client = self._http()
        resp = await client.post(
            f"{self._registry_url}/v1/skills/call",
            json={"skill": tool_name, "arguments": arguments},
            headers=headers,
            timeout=60.0,
        )

        if resp.status_code >= 400:
            raise ToolError(
                f"Skill '{tool_name}' returned HTTP {resp.status_code}. "
                f"Arguments: {arguments}. "
                f"Response: {resp.text[:200]}. "
                "Fix: check the skill name, arguments, and registry API key."
            )

        return resp.json().get("result", "")
