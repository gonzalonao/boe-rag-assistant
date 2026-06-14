"""Tests for the LLM provider layer, using mocked transports and a fake provider."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from boe_rag.llm.base import ChatMessage, LLMError, LLMRateLimitError
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


def test_provider_raises_rate_limit_error_after_retries() -> None:
    """A persistent 429 is retried then surfaced as LLMRateLimitError."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # retry-after: 0 keeps the test fast while exercising the header path.
        return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")

    provider = GroqProvider(api_key="k")
    _mock(provider, handler)
    with pytest.raises(LLMRateLimitError):
        provider.complete([ChatMessage(role="user", content="q")])
    assert calls["n"] == 4


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


class _CountingProvider:
    """Provider that counts calls and optionally raises a fixed error."""

    def __init__(self, name: str, *, error: Exception | None) -> None:
        self._name = name
        self._error = error
        self.calls = 0

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
        self.calls += 1
        if self._error is not None:
            raise self._error
        return "ok"


def test_fallback_trips_breaker_on_rate_limit() -> None:
    """A rate-limited provider is skipped on subsequent calls, not retried."""
    limited = _CountingProvider("a", error=LLMRateLimitError("a 429"))
    healthy = _CountingProvider("b", error=None)
    chain = FallbackProvider([limited, healthy])
    msg = [ChatMessage(role="user", content="q")]

    assert chain.complete(msg) == "ok"
    assert chain.complete(msg) == "ok"
    assert limited.calls == 1  # tripped after the first 429
    assert healthy.calls == 2


def test_fallback_raises_rate_limit_error_when_all_tripped() -> None:
    """Once the only provider is cooling down, the chain reports rate-limited."""
    limited = _CountingProvider("a", error=LLMRateLimitError("429"))
    chain = FallbackProvider([limited])
    msg = [ChatMessage(role="user", content="q")]

    with pytest.raises(LLMRateLimitError):
        chain.complete(msg)
    # Still within the cool-down window: the provider is skipped, not re-tried.
    with pytest.raises(LLMRateLimitError, match="rate-limited"):
        chain.complete(msg)
    assert limited.calls == 1


def test_fallback_retries_provider_after_cooldown() -> None:
    """With a zero cool-down, a rate-limited provider is retried, not disabled."""
    limited = _CountingProvider("a", error=LLMRateLimitError("429"))
    chain = FallbackProvider([limited], cooldown=0.0)
    msg = [ChatMessage(role="user", content="q")]

    with pytest.raises(LLMRateLimitError):
        chain.complete(msg)
    with pytest.raises(LLMRateLimitError):
        chain.complete(msg)
    assert limited.calls == 2  # recovered after cool-down, tried again


def test_build_available_providers_skips_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only providers with a configured key are constructed."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    providers = build_available_providers()
    assert [p.name.split(":")[0] for p in providers] == ["groq"]
