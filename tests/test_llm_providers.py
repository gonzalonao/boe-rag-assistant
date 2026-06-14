"""Tests for the LLM provider layer, using mocked transports and a fake provider."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from boe_rag.llm.base import ChatMessage, LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.llm.gemini import GeminiProvider
from boe_rag.llm.groq import GroqProvider


def _mock(provider: GeminiProvider | GroqProvider, handler: object) -> None:
    """Swap a provider's HTTP client for one driven by a mock handler."""
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_gemini_parses_completion() -> None:
    """GeminiProvider extracts text from a generateContent response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "k"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Hola mundo"}]}}]},
        )

    provider = GeminiProvider(api_key="k")
    _mock(provider, handler)
    out = provider.complete([ChatMessage(role="user", content="hi")])
    assert out == "Hola mundo"


def test_gemini_sends_system_instruction() -> None:
    """A system message is mapped to Gemini's systemInstruction field."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )

    provider = GeminiProvider(api_key="k")
    _mock(provider, handler)
    provider.complete(
        [
            ChatMessage(role="system", content="Eres un juez."),
            ChatMessage(role="user", content="evalúa"),
        ]
    )
    assert "systemInstruction" in seen


def test_groq_parses_completion() -> None:
    """GroqProvider extracts text from an OpenAI-style response."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "respuesta"}}]}
        )

    provider = GroqProvider(api_key="k")
    _mock(provider, handler)
    assert provider.complete([ChatMessage(role="user", content="q")]) == "respuesta"


def test_provider_retries_then_succeeds() -> None:
    """A transient 429 is retried until success."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = GroqProvider(api_key="k")
    _mock(provider, handler)
    assert provider.complete([ChatMessage(role="user", content="q")]) == "ok"
    assert calls["n"] == 2


def test_provider_raises_on_client_error() -> None:
    """A 400 surfaces as an LLMError without retrying."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    provider = GroqProvider(api_key="k")
    _mock(provider, handler)
    with pytest.raises(LLMError):
        provider.complete([ChatMessage(role="user", content="q")])


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a provider without a key is an error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqProvider()


class _FakeProvider:
    """Minimal provider for fallback tests."""

    def __init__(self, name: str, *, fail: bool, reply: str = "") -> None:
        self._name = name
        self._fail = fail
        self._reply = reply

    @property
    def name(self) -> str:
        return self._name

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self._fail:
            raise LLMError(f"{self._name} down")
        return self._reply


def test_fallback_uses_second_when_first_fails() -> None:
    """The fallback chain skips a failing provider and uses the next."""
    chain = FallbackProvider(
        [_FakeProvider("a", fail=True), _FakeProvider("b", fail=False, reply="ok")]
    )
    assert chain.complete([ChatMessage(role="user", content="q")]) == "ok"


def test_fallback_raises_when_all_fail() -> None:
    """If every provider fails, the chain raises a combined error."""
    chain = FallbackProvider(
        [_FakeProvider("a", fail=True), _FakeProvider("b", fail=True)]
    )
    with pytest.raises(LLMError, match="all providers failed"):
        chain.complete([ChatMessage(role="user", content="q")])


def test_build_available_providers_skips_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only providers with a configured key are constructed."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    providers = build_available_providers()
    assert [p.name.split(":")[0] for p in providers] == ["groq"]
