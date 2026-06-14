"""Groq chat-completion provider via its OpenAI-compatible REST API.

Reads the API key from ``GROQ_API_KEY`` unless one is passed explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from boe_rag.llm._http import post_json_with_retry
from boe_rag.llm.base import ChatMessage, LLMError

_URL = "https://api.groq.com/openai/v1/chat/completions"

#: Default free-tier model: capable and fast for judging and generation.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqProvider:
    """Groq chat-completion provider (OpenAI-compatible API).

    Args:
        api_key: API key; falls back to ``GROQ_API_KEY``.
        model: Groq model id.
        timeout: Per-request timeout in seconds.

    Raises:
        LLMError: If no API key can be found.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Resolve the API key and open an HTTP client."""
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("Groq API key not found; set GROQ_API_KEY.")
        self._key = key
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        """Provider identifier including the model id."""
        return f"groq:{self._model}"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion via the Groq chat-completions endpoint."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = post_json_with_retry(
            self._client,
            _URL,
            json=body,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=self._timeout,
        )
        return _extract_text(data)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the generated text out of a Groq response payload."""
    try:
        content = data["choices"][0]["message"]["content"]
        text: str = content.strip()
    except (KeyError, IndexError, TypeError, AttributeError) as err:
        raise LLMError(f"unexpected Groq response shape: {data}") from err
    if not text:
        raise LLMError(f"empty Groq completion: {data}")
    return text
