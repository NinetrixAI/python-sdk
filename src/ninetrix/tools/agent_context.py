"""
tools/agent_context.py — shared services injected into every ToolSource.

Layer: L2 (tools) — imports L1 (_internals) + stdlib only.

AgentContext is created once per agent and passed to all ToolSource instances.
It provides shared HTTP, auth, logging, and environment access so that sources
don't each reinvent these concerns.

Not to be confused with :class:`~ninetrix.tools.context.ToolContext`, which is
per-call injection into individual ``@Tool`` functions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentContext:
    """Shared services available to all :class:`~ninetrix.ToolSource` instances.

    Constructed by :meth:`Agent._build_runner()` and passed to each source's
    ``__init__``.  Sources should prefer these shared services over creating
    their own HTTP clients or reading env vars directly.

    Attributes:
        http:        Process-wide ``httpx.AsyncClient`` singleton.
        env:         Callable ``(key, default) -> value`` for reading env vars.
                     Defaults to ``os.environ.get``.  Abstracted for testing.
        logger:      Structured logger scoped to the agent.
        auth:        :class:`~ninetrix.tools.auth_resolver.AuthResolver` for
                     converting ``agentfile.yaml`` auth blocks into HTTP headers.
        agent_name:  Name of the owning agent (for logging / telemetry).
        org_id:      Organization ID for multi-tenant contexts.
    """

    http: Any  # httpx.AsyncClient — typed as Any to avoid L1 import at class level
    env: Callable[..., str] = field(default=os.environ.get)
    logger: Any = field(default_factory=lambda: logging.getLogger("ninetrix.agent"))
    auth: Any = None  # AuthResolver — typed as Any to avoid circular import
    agent_name: str = ""
    org_id: str = ""
