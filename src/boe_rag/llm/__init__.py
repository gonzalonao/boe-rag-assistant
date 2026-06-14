"""Provider-agnostic LLM access (Gemini, Groq) with a fallback chain."""

from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider
from boe_rag.llm.factory import (
    FallbackProvider,
    build_available_providers,
    build_provider,
)
from boe_rag.llm.gemini import GeminiProvider
from boe_rag.llm.groq import GroqProvider

__all__ = [
    "ChatMessage",
    "FallbackProvider",
    "GeminiProvider",
    "GroqProvider",
    "LLMError",
    "LLMProvider",
    "build_available_providers",
    "build_provider",
]
