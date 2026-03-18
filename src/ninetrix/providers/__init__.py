"""
ninetrix.providers
==================

LLM provider adapters — normalize any LLM API into LLMResponse.

Adapters are optional dependencies; import only what you need:

    from ninetrix.providers.anthropic import AnthropicAdapter
    from ninetrix.providers.openai import OpenAIAdapter
    from ninetrix.providers.google import GoogleAdapter
    from ninetrix.providers.litellm import LiteLLMAdapter
    from ninetrix.providers.fallback import FallbackConfig, FallbackProviderAdapter

The Protocol all adapters satisfy is re-exported from providers.base:

    from ninetrix.providers.base import LLMProviderAdapter
"""

from ninetrix.providers.base import LLMProviderAdapter as LLMProviderAdapter
from ninetrix.providers.fallback import (
    FallbackConfig as FallbackConfig,
    FallbackProviderAdapter as FallbackProviderAdapter,
)

__all__ = [
    "LLMProviderAdapter",
    "FallbackConfig",
    "FallbackProviderAdapter",
]
