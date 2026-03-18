"""
Ninetrix SDK — Python primitives for building AI agent tools.

Phase 1: @Tool decorator
    Define Python functions as agent-callable tools. They are auto-discovered
    by ``ninetrix build``, bundled into the Docker image, and dispatched at
    runtime alongside MCP tools.

Phase 2 (upcoming): Agent + Workflow
    Define full agents and complex workflows in Python.

Quick start::

    from ninetrix import Tool

    @Tool
    def query_customers(sql: str, limit: int = 100) -> list[dict]:
        \"\"\"Run a read-only SQL query against the customer database.

        Args:
            sql: A valid SELECT statement.
            limit: Maximum rows to return.
        \"\"\"
        return db.execute(sql, limit=limit)

Reference the tool in ``agentfile.yaml``::

    tools:
      - name: query_customers
        source: ./tools/db_tools.py

Then build and run as usual::

    ninetrix build
    ninetrix run
"""

from ninetrix.tool import Tool as Tool
from ninetrix.registry import ToolDef as ToolDef, ToolRegistry as ToolRegistry, _registry
from ninetrix.discover import (
    discover_tools_in_file as discover_tools_in_file,
    discover_tools_in_files as discover_tools_in_files,
    load_local_tools as load_local_tools,
)
from ninetrix._internals.lifespan import (
    startup as startup,
    shutdown as shutdown,
    lifespan as lifespan,
)
from ninetrix.providers import (
    FallbackConfig as FallbackConfig,
    FallbackProviderAdapter as FallbackProviderAdapter,
)
from ninetrix.observability.logger import (
    NinetrixLogger as NinetrixLogger,
    enable_debug as enable_debug,
    get_logger as get_logger,
)
from ninetrix.observability.telemetry import (
    TelemetryEvent as TelemetryEvent,
    TelemetryCollector as TelemetryCollector,
    record_event as record_event,
)
from ninetrix.observability.errors import (
    ErrorContext as ErrorContext,
    error_context as error_context,
)
from ninetrix._internals.tenant import (
    TenantContext as TenantContext,
    set_tenant as set_tenant,
    get_tenant as get_tenant,
    require_tenant as require_tenant,
    tenant_scope as tenant_scope,
)
from ninetrix.runtime.history import MessageHistory as MessageHistory
from ninetrix.runtime.budget import BudgetTracker as BudgetTracker, BudgetUsage as BudgetUsage
from ninetrix.runtime.runner import AgentRunner as AgentRunner, RunnerConfig as RunnerConfig
from ninetrix.checkpoint.base import Checkpointer as Checkpointer
from ninetrix.checkpoint.memory import InMemoryCheckpointer as InMemoryCheckpointer
from ninetrix.agent.config import AgentConfig as AgentConfig
from ninetrix.agent.introspection import (
    AgentInfo as AgentInfo,
    ValidationIssue as ValidationIssue,
    DryRunResult as DryRunResult,
)
from ninetrix.agent.agent import Agent as Agent
from ninetrix.export.writer import agent_to_yaml as agent_to_yaml
from ninetrix.export.loader import (
    load_agent_from_yaml as load_agent_from_yaml,
    load_all_agents_from_yaml as load_all_agents_from_yaml,
)
from ninetrix.runtime.dispatcher import (
    ToolSource as ToolSource,
    ToolDispatcher as ToolDispatcher,
    LocalToolSource as LocalToolSource,
    RegistryToolSource as RegistryToolSource,
)
from ninetrix.tools.context import ToolContext as ToolContext
from ninetrix._internals.types import (
    # Result / event types
    AgentResult as AgentResult,
    StreamEvent as StreamEvent,
    WorkflowResult as WorkflowResult,
    StepResult as StepResult,
    # Attachment helpers
    ImageAttachment as ImageAttachment,
    DocumentAttachment as DocumentAttachment,
    Attachment as Attachment,
    image as image,
    document as document,
    # Provider config
    ProviderConfig as ProviderConfig,
    # Protocols
    AgentProtocol as AgentProtocol,
    # Errors
    NinetrixError as NinetrixError,
    CredentialError as CredentialError,
    ProviderError as ProviderError,
    ToolError as ToolError,
    BudgetExceededError as BudgetExceededError,
    OutputParseError as OutputParseError,
    CheckpointError as CheckpointError,
    ApprovalTimeoutError as ApprovalTimeoutError,
    ConfigurationError as ConfigurationError,
    NetworkError as NetworkError,
)

__version__ = "0.1.0"
__all__ = [
    # PR 19 — YAML round-trip
    "agent_to_yaml",
    "load_agent_from_yaml",
    "load_all_agents_from_yaml",
    # PR 18 — Agent + AgentConfig + introspection
    "Agent",
    "AgentConfig",
    "AgentInfo",
    "ValidationIssue",
    "DryRunResult",
    # PR 17 — Checkpointer + InMemoryCheckpointer
    "Checkpointer",
    "InMemoryCheckpointer",
    # PR 16 — AgentRunner + RunnerConfig
    "AgentRunner",
    "RunnerConfig",
    # PR 15 — ToolDispatcher + ToolContext
    "ToolSource",
    "ToolDispatcher",
    "LocalToolSource",
    "RegistryToolSource",
    "ToolContext",
    # PR 14 — MessageHistory + BudgetTracker
    "MessageHistory",
    "BudgetTracker",
    "BudgetUsage",
    # PR 13 — TenantContext
    "TenantContext",
    "set_tenant",
    "get_tenant",
    "require_tenant",
    "tenant_scope",
    # PR 9 — ErrorContext
    "ErrorContext",
    "error_context",
    # PR 8 — Telemetry
    "TelemetryEvent",
    "TelemetryCollector",
    "record_event",
    # PR 7 — Logger
    "NinetrixLogger",
    "enable_debug",
    "get_logger",
    # PR 6 — Providers
    "FallbackConfig",
    "FallbackProviderAdapter",
    # Phase 1 — @Tool decorator
    "Tool",
    "ToolDef",
    "ToolRegistry",
    "discover_tools_in_file",
    "discover_tools_in_files",
    "load_local_tools",
    "_registry",
    # PR 5 — lifespan
    "startup",
    "shutdown",
    "lifespan",
    # PR 1 — types
    "AgentResult",
    "StreamEvent",
    "WorkflowResult",
    "StepResult",
    "ImageAttachment",
    "DocumentAttachment",
    "Attachment",
    "image",
    "document",
    "ProviderConfig",
    "AgentProtocol",
    "NinetrixError",
    "CredentialError",
    "ProviderError",
    "ToolError",
    "BudgetExceededError",
    "OutputParseError",
    "CheckpointError",
    "ApprovalTimeoutError",
    "ConfigurationError",
    "NetworkError",
]
