"""Tests for the RagEngine pipeline and its observability tracing seam."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from boe_rag.llm.base import ChatMessage
from boe_rag.service.engine import ChunkInfo, RagEngine
from boe_rag.service.tracing import NoOpTracer, Span


class _FakeSearcher:
    """Returns a fixed ranked list, ignoring the query."""

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._results[:k]


class _ReverseReranker:
    """Reranker that simply reverses the candidate order (deterministic)."""

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        return [(cid, float(i)) for i, (cid, _) in enumerate(reversed(candidates))]


class _FakeProvider:
    """LLM provider returning a canned grounded answer."""

    @property
    def name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        return "El tipo general del IVA es del 21% [1]."


@dataclass
class _RecordingSpan:
    """A span that records the updates it receives."""

    name: str
    outputs: list[object] = field(default_factory=list)
    metadata: list[Mapping[str, object]] = field(default_factory=list)

    def update(
        self,
        *,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if output is not None:
            self.outputs.append(output)
        if metadata is not None:
            self.metadata.append(metadata)


class _RecordingTracer:
    """Tracer that records every span opened, in order."""

    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    @contextmanager
    def span(self, name: str, **inputs: object) -> Iterator[Span]:
        recorded = _RecordingSpan(name=name)
        self.spans.append(recorded)
        yield recorded

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.spans]


_LOOKUP: Mapping[str, ChunkInfo] = {
    "c1": ChunkInfo(citation="Ley 1/2024, Art. 1", text="IVA general 21%.", url="u1"),
    "c2": ChunkInfo(citation="Ley 2/2024, Art. 2", text="IVA reducido 10%.", url="u2"),
}


def _engine(
    tracer: _RecordingTracer | None = None, *, rerank: bool = True
) -> RagEngine:
    return RagEngine(
        retriever=_FakeSearcher([("c1", 0.9), ("c2", 0.5)]),
        lookup=_LOOKUP,
        provider=_FakeProvider(),
        reranker=_ReverseReranker() if rerank else None,
        tracer=tracer,
    )


def test_engine_runs_without_a_tracer() -> None:
    """The default no-op tracer leaves behaviour unchanged."""
    engine = _engine(rerank=False)
    response = engine.answer("¿IVA?", k=2)
    assert "21%" in response.answer
    assert not response.refused
    assert [s.chunk_id for s in response.sources] == ["c1", "c2"]


def test_noop_tracer_span_is_usable() -> None:
    """NoOpTracer yields a span whose update is a harmless no-op."""
    with NoOpTracer().span("x", a=1) as span:
        span.update(output="ignored", metadata={"k": 1})


def test_answer_opens_a_span_per_stage() -> None:
    """answer() instruments retrieve, rerank, generate, and the root answer span."""
    tracer = _RecordingTracer()
    _engine(tracer).answer("¿IVA?", k=2)
    assert tracer.names == ["answer", "retrieve", "rerank", "generate"]


def test_generate_span_captures_the_answer_text() -> None:
    """The generate span records the produced answer as its output."""
    tracer = _RecordingTracer()
    _engine(tracer).answer("¿IVA?", k=2)
    generate = next(s for s in tracer.spans if s.name == "generate")
    assert generate.outputs == ["El tipo general del IVA es del 21% [1]."]


def test_search_traces_retrieve_and_rerank_only() -> None:
    """search() (no generation) instruments just the retrieval stages."""
    tracer = _RecordingTracer()
    _engine(tracer).search("¿IVA?", k=2)
    assert tracer.names == ["retrieve", "rerank"]


def test_rerank_reorders_sources() -> None:
    """The reranker's order is reflected in the returned sources."""
    sources = _engine(rerank=True).search("¿IVA?", k=2)
    assert [s.chunk_id for s in sources] == ["c2", "c1"]
