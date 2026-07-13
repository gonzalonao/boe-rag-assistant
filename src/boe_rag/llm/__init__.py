"""Provider-agnostic LLM access (OpenRouter, Groq) with a fallback chain."""

from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider
from boe_rag.llm.factory import (
    FallbackProvider,
    build_available_providers,
    build_provider,
)
from boe_rag.llm.groq import GroqProvider
from boe_rag.llm.openrouter import OpenRouterProvider

__all__ = [
    "ChatMessage",
    "FallbackProvider",
    "GroqProvider",
    "LLMError",
    "LLMProvider",
    "OpenRouterProvider",
    "build_available_providers",
    "build_provider",
]
