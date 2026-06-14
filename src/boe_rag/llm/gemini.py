"""Google AI Studio (Gemini) chat-completion provider via the REST API.

Uses the public ``generativelanguage`` REST endpoint over httpx (no heavy
vendor SDK). Reads the API key from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
unless one is passed explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from boe_rag.llm._http import post_json_with_retry
from boe_rag.llm.base import ChatMessage, LLMError

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

#: Default free-tier model: fast, multilingual, strong on Spanish.
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiProvider:
    """Gemini chat-completion provider.

    Args:
        api_key: API key; falls back to ``GEMINI_API_KEY``/``GOOGLE_API_KEY``.
        model: Gemini model id.
        timeout: Per-request timeout in seconds.

    Raises:
        LLMError: If no API key can be found.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Resolve the API key and open an HTTP client."""
        key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise LLMError(
                "Gemini API key not found; set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self._key = key
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        """Provider identifier including the model id."""
        return f"gemini:{self._model}"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion via the Gemini ``generateContent`` endpoint."""
        system_text = " ".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = f"{_BASE_URL}/{self._model}:generateContent"
        data = post_json_with_retry(
            self._client,
            url,
            json=body,
            headers={"x-goog-api-key": self._key},
            timeout=self._timeout,
        )
        return _extract_text(data)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the generated text out of a Gemini response payload."""
    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as err:
        raise LLMError(f"unexpected Gemini response shape: {data}") from err
    if not text:
        raise LLMError(f"empty Gemini completion: {data}")
    return text
