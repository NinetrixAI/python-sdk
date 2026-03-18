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
