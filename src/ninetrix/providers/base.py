"""
ninetrix.providers.base
=======================

Re-exports ``LLMProviderAdapter`` from ``_internals.types`` — the stable
public import point for the Protocol.

Concrete adapters live in their own files (anthropic.py, openai.py, etc.)
and all inherit from this Protocol so that::

    issubclass(AnthropicAdapter, LLMProviderAdapter)  # True
    isinstance(adapter, LLMProviderAdapter)           # True
"""

from ninetrix._internals.types import LLMProviderAdapter as LLMProviderAdapter

__all__ = ["LLMProviderAdapter"]
