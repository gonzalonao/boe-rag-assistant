"""OpenRouter chat-completion provider via its OpenAI-compatible REST API.

OpenRouter brokers many models behind one OpenAI-style endpoint, including a
catalogue of ``:free`` models. With a small prepaid credit balance the free tier
allows a larger daily request budget; the per-minute and per-day caps both
surface as HTTP 429, so the fallback chain's cool-down naturally rides OpenRouter
until it is exhausted and only then moves on to the next provider.

Reads the API key from ``OPENROUTER_API_KEY`` unless one is passed explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from boe_rag.llm._http import post_json_with_retry
from boe_rag.llm.base import ChatMessage, LLMError

_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Default free model: capable and multilingual (good for Spanish), no token
#: cost on the free tier. Override with ``OPENROUTER_MODEL``; browse the current
#: free catalogue at https://openrouter.ai/models?max_price=0.
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

#: Sent as OpenRouter's optional ranking headers; harmless and identifies the app.
_APP_URL = "https://github.com/gonzalonao/boe-rag-assistant"
_APP_TITLE = "BOE RAG Assistant"


class OpenRouterProvider:
    """OpenRouter chat-completion provider (OpenAI-compatible API).

    Args:
        api_key: API key; falls back to ``OPENROUTER_API_KEY``.
        model: OpenRouter model id; falls back to ``OPENROUTER_MODEL`` then the
            default free model. Append ``:free`` to use a model's free variant.
        timeout: Per-request timeout in seconds.

    Raises:
        LLMError: If no API key can be found.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Resolve the API key and open an HTTP client."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OpenRouter API key not found; set OPENROUTER_API_KEY.")
        self._key = key
        self._model = (
            model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        )
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        """Provider identifier including the model id."""
        return f"openrouter:{self._model}"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion via the OpenRouter chat-completions endpoint."""
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
            headers={
                "Authorization": f"Bearer {self._key}",
                "HTTP-Referer": _APP_URL,
                "X-Title": _APP_TITLE,
            },
            timeout=self._timeout,
        )
        return _extract_text(data)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the generated text out of an OpenRouter response payload."""
    try:
        content = data["choices"][0]["message"]["content"]
        text: str = content.strip()
    except (KeyError, IndexError, TypeError, AttributeError) as err:
        raise LLMError(f"unexpected OpenRouter response shape: {data}") from err
    if not text:
        raise LLMError(f"empty OpenRouter completion: {data}")
    return text
