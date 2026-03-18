"""
observability/streaming.py — streaming event types and callback protocols.

Layer: L6 (observability) — may import L1 (_internals) + stdlib only.

The agentic streaming runner lives in runtime/streaming.py (L3), which sits
above this layer.  This module holds only the observable surface:
- StreamEvent is defined in _internals/types.py (L1)
- StreamingRunner is exported from ninetrix.__init__ via runtime/streaming.py

Re-exported here for ergonomic discovery:
    from ninetrix.observability.streaming import StreamEvent
"""

from ninetrix._internals.types import StreamEvent as StreamEvent  # noqa: F401

__all__ = ["StreamEvent"]
