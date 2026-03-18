"""
ninetrix.observability.errors
==============================
L6 kernel — stdlib only.

``ErrorContext`` enriches ``NinetrixError`` exceptions with structured
context (agent name, thread ID, step, etc.) before they propagate up to
the caller.

The context is stored directly on the exception as ``exc._context``, so it
survives ``raise ... from exc`` chains and can be read by logging handlers,
middleware, and user error handlers.

Usage (inside the runner or agent layer)::

    from ninetrix.observability.errors import ErrorContext

    async def _run_step(self, ...):
        with ErrorContext(agent_name="analyst", thread_id="t-123", step=3):
            result = await provider.complete(...)   # may raise ProviderError

    # ProviderError.context → {"agent_name": "analyst", "thread_id": "t-123", "step": 3}

You can also attach context to an already-caught exception::

    try:
        await tool_source.call(name, args)
    except NinetrixError as exc:
        ErrorContext.attach(exc, agent_name="analyst", thread_id=thread_id)
        raise

The context dict is merged on each attach, so nested ``with`` blocks
accumulate fields (inner wins on duplicate keys).
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import Any, Generator

from ninetrix._internals.types import NinetrixError

# Attribute name stored on the exception
_CTX_ATTR = "_context"


class ErrorContext:
    """
    Context manager that attaches structured context to any ``NinetrixError``
    raised inside the ``with`` block.

    Fields passed as kwargs are merged into ``exc._context``.  Nested
    ``ErrorContext`` blocks accumulate — each layer adds its own fields.

    Example::

        with ErrorContext(agent_name="analyst", thread_id="t-99"):
            with ErrorContext(step=4, tool="search"):
                raise ProviderError("timeout")
        # ProviderError._context == {
        #     "agent_name": "analyst", "thread_id": "t-99",
        #     "step": 4, "tool": "search"
        # }

    Also usable as a static helper::

        ErrorContext.attach(exc, agent_name="x", step=2)
    """

    def __init__(self, **ctx: Any) -> None:
        self._ctx = ctx

    def __enter__(self) -> "ErrorContext":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        if exc_val is not None and isinstance(exc_val, NinetrixError):
            self._merge(exc_val, self._ctx)
        return False  # never suppress — always re-raise

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def attach(exc: NinetrixError, **ctx: Any) -> None:
        """
        Attach context fields to an already-caught ``NinetrixError``.

        Merges into any existing ``_context`` dict on the exception.
        New keys win over existing ones.

        Example::

            except NinetrixError as exc:
                ErrorContext.attach(exc, agent_name="x", thread_id=tid)
                raise
        """
        ErrorContext._merge(exc, ctx)

    @staticmethod
    def get(exc: BaseException) -> dict[str, Any]:
        """
        Return the context dict attached to an exception (empty dict if none).

        Works on any exception — returns ``{}`` for non-NinetrixError types.
        """
        return dict(getattr(exc, _CTX_ATTR, {}))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(exc: NinetrixError, ctx: dict[str, Any]) -> None:
        existing: dict[str, Any] = getattr(exc, _CTX_ATTR, {})
        merged = {**existing, **ctx}   # new keys win
        object.__setattr__(exc, _CTX_ATTR, merged) if False else setattr(exc, _CTX_ATTR, merged)


# ---------------------------------------------------------------------------
# Convenience: context manager that works on any NinetrixError subclass
# ---------------------------------------------------------------------------

@contextmanager
def error_context(**ctx: Any) -> Generator[None, None, None]:
    """
    Functional form of ``ErrorContext`` — useful when you don't need the
    ``ErrorContext`` object itself::

        async with error_context(agent_name="analyst", thread_id=tid):
            ...
    """
    with ErrorContext(**ctx):
        yield
