"""Tests for the Langfuse tracing adapter and its factory.

The adapter is exercised with a fake Langfuse client, so these tests need no
``langfuse`` install and no network — they pin the adapter's contract: it opens a
span per stage, records inputs, and forwards only the fields it is given.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from boe_rag.llm.base import ChatMessage
from boe_rag.service.engine import ChunkInfo, RagEngine
from boe_rag.service.tracing import (
    LangfuseTracer,
    NoOpTracer,
    Tracer,
    _LangfuseSpan,
    build_tracer,
)


@dataclass
class _FakeLangfuseSpan:
    """Records the keyword updates forwarded to it."""

    updates: list[dict[str, object]] = field(default_factory=list)

    def update(self, **kwargs: object) -> None:
        """Record one forwarded update."""
        self.updates.append(kwargs)


class _FakeLangfuseClient:
    """A stand-in for the Langfuse v3 client used by the adapter."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, object]] = []
        self.handles: list[_FakeLangfuseSpan] = []
        self.trace_names: list[str] = []

    @contextmanager
    def start_as_current_span(
        self, *, name: str, input: object = None
    ) -> Iterator[_FakeLangfuseSpan]:
        """Open a fake span, recording its name and inputs."""
        handle = _FakeLangfuseSpan()
        self.opened.append((name, input))
        self.handles.append(handle)
        yield handle

    def update_current_trace(self, *, name: str) -> None:
        """Record the name set on the current trace."""
        self.trace_names.append(name)


def test_langfuse_tracer_is_a_tracer() -> None:
    """The adapter structurally satisfies the Tracer protocol."""
    assert isinstance(LangfuseTracer(_FakeLangfuseClient()), Tracer)


def test_span_records_name_and_inputs() -> None:
    """Opening a span passes the stage name and inputs to Langfuse."""
    client = _FakeLangfuseClient()
    with LangfuseTracer(client).span("retrieve", query="¿IVA?", k=5):
        pass
    assert client.opened == [("retrieve", {"query": "¿IVA?", "k": 5})]


def test_span_with_query_names_the_trace() -> None:
    """A stage carrying a query names the trace after it (searchable, clean tree)."""
    client = _FakeLangfuseClient()
    with LangfuseTracer(client).span("answer", query="¿IVA?", k=5):
        pass
    assert client.trace_names == ["¿IVA?"]


def test_span_without_query_leaves_the_trace_name_unset() -> None:
    """Stages without a query (e.g. generate) do not touch the trace name."""
    client = _FakeLangfuseClient()
    with LangfuseTracer(client).span("generate", contexts=3):
        pass
    assert client.trace_names == []


def test_span_update_forwards_only_provided_fields() -> None:
    """``update`` forwards output and metadata only when they are given."""
    handle = _FakeLangfuseSpan()
    span = _LangfuseSpan(handle)
    span.update()
    span.update(output="text")
    span.update(metadata={"n": 1})
    span.update(output="t", metadata={"n": 2})
    assert handle.updates == [
        {"output": "text"},
        {"metadata": {"n": 1}},
        {"output": "t", "metadata": {"n": 2}},
    ]


def test_build_tracer_is_noop_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Langfuse keys means a no-op tracer."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert isinstance(build_tracer(), NoOpTracer)


def test_build_tracer_is_noop_with_partial_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single key is not enough; the tracer stays a no-op."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert isinstance(build_tracer(), NoOpTracer)


def test_build_tracer_falls_back_when_langfuse_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With keys set but no ``langfuse`` installed, fall back to a no-op."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    try:
        import langfuse  # noqa: F401
    except ImportError:
        assert isinstance(build_tracer(), NoOpTracer)
    else:
        pytest.skip("langfuse is installed; the fallback path is not exercised")


# --- Engine integration: the adapter must work as the engine's real tracer. ---


class _FakeSearcher:
    """Returns a fixed ranked list, ignoring the query."""

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``k`` canned results."""
        return [("c1", 0.9), ("c2", 0.5)][:k]


class _ReverseReranker:
    """Reranker that reverses the candidate order deterministically."""

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Reverse the candidates and assign descending scores."""
        return [(cid, float(i)) for i, (cid, _) in enumerate(reversed(candidates))]


class _FakeProvider:
    """LLM provider returning a canned grounded answer."""

    @property
    def name(self) -> str:
        """Provider name."""
        return "fake"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return a fixed grounded answer."""
        return "El tipo general del IVA es del 21% [1]."


_LOOKUP = {
    "c1": ChunkInfo(citation="Ley 1/2024, Art. 1", text="IVA general 21%.", url="u1"),
    "c2": ChunkInfo(citation="Ley 2/2024, Art. 2", text="IVA reducido 10%.", url="u2"),
}


def test_engine_traces_full_pipeline_through_langfuse_adapter() -> None:
    """Driving the real engine opens the expected nested Langfuse spans."""
    client = _FakeLangfuseClient()
    engine = RagEngine(
        retriever=_FakeSearcher(),
        lookup=_LOOKUP,
        provider=_FakeProvider(),
        reranker=_ReverseReranker(),
        tracer=LangfuseTracer(client),
    )
    engine.answer("¿IVA?", k=2)
    assert [name for name, _ in client.opened] == [
        "answer",
        "retrieve",
        "rerank",
        "generate",
    ]
    # The generate span received the produced answer as its output.
    generate_handle = client.handles[3]
    assert any("output" in u for u in generate_handle.updates)
