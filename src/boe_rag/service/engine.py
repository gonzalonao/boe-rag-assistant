"""The RAG engine: the query-time pipeline behind the service.

Composes the retrieval and generation pieces built and measured in earlier
phases into a single object: retrieve a candidate pool, optionally rerank it with
the cross-encoder, then generate a grounded, cited answer (or refuse). The
service depends on the :class:`Engine` protocol, so the FastAPI layer can be
tested with a trivial fake and the heavy models stay out of CI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from boe_rag.eval.answerer import REFUSAL, SYSTEM_PROMPT_CANARY, generate_answer
from boe_rag.eval.rerank import DEFAULT_RERANK_POOL, Reranker
from boe_rag.eval.retriever import Searcher
from boe_rag.llm.base import LLMProvider
from boe_rag.service.citation import validate_citations
from boe_rag.service.models import AnswerResponse, Source
from boe_rag.service.safety import screen_canary
from boe_rag.service.tracing import NoOpTracer, Tracer


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    """Everything the engine needs to cite a chunk.

    Attributes:
        citation: Human-readable citation (e.g. law and article).
        text: The chunk text.
        url: Link back to the source document on boe.es.
    """

    citation: str
    text: str
    url: str


class Engine(Protocol):
    """The query-time interface the API depends on."""

    @property
    def num_chunks(self) -> int:
        """Number of indexed corpus chunks."""
        ...

    def search(self, query: str, k: int = 10) -> list[Source]:
        """Retrieve the top-k passages for a query."""
        ...

    def answer(self, query: str, k: int = 5) -> AnswerResponse:
        """Answer a question, grounded in the retrieved passages."""
        ...


class RagEngine:
    """Two-stage retrieval plus grounded generation.

    Args:
        retriever: First-stage retriever (e.g. hybrid).
        lookup: Maps a chunk id to its citation, text, and URL.
        provider: LLM provider for answer generation.
        reranker: Optional cross-encoder reranker for a second stage.
        rerank_pool: First-stage candidates to rerank when a reranker is set.
        tracer: Observability tracer for per-stage spans; defaults to a no-op so
            the engine runs identically until a backend (e.g. Langfuse) is wired.
    """

    def __init__(
        self,
        retriever: Searcher,
        lookup: Mapping[str, ChunkInfo],
        provider: LLMProvider,
        reranker: Reranker | None = None,
        rerank_pool: int = DEFAULT_RERANK_POOL,
        tracer: Tracer | None = None,
    ) -> None:
        """Bind the pipeline stages and the chunk lookup."""
        self._retriever = retriever
        self._lookup = lookup
        self._provider = provider
        self._reranker = reranker
        self._rerank_pool = rerank_pool
        self._tracer = tracer or NoOpTracer()

    @property
    def num_chunks(self) -> int:
        """Number of indexed corpus chunks."""
        return len(self._lookup)

    def _ranked_chunks(self, query: str, k: int) -> list[tuple[str, float]]:
        """Retrieve (and optionally rerank) the top-k ``(chunk_id, score)``."""
        with self._tracer.span("retrieve", query=query, k=k) as retrieval:
            if self._reranker is None:
                results = self._retriever.search(query, k)
                retrieval.update(metadata={"n": len(results)})
                return results
            pool = max(self._rerank_pool, k)
            candidates = self._retriever.search(query, pool)
            retrieval.update(metadata={"pool": pool, "candidates": len(candidates)})
        pairs = [
            (cid, self._lookup[cid].text)
            for cid, _ in candidates
            if cid in self._lookup
        ]
        if not pairs:
            return []
        with self._tracer.span("rerank", candidates=len(pairs), k=k) as rerank:
            ranked = self._reranker.rerank(query, pairs)[:k]
            rerank.update(metadata={"n": len(ranked)})
        return ranked

    def search(self, query: str, k: int = 10) -> list[Source]:
        """Retrieve the top-k passages for a query.

        Args:
            query: The search query.
            k: Maximum number of passages to return.

        Returns:
            The retrieved passages as :class:`Source` objects, best first.
        """
        sources: list[Source] = []
        for chunk_id, score in self._ranked_chunks(query, k):
            info = self._lookup.get(chunk_id)
            if info is None:
                continue
            sources.append(
                Source(
                    chunk_id=chunk_id,
                    citation=info.citation,
                    text=info.text,
                    url=info.url,
                    score=score,
                )
            )
        return sources

    def answer(self, query: str, k: int = 5) -> AnswerResponse:
        """Answer a question, grounded in the retrieved passages.

        Args:
            query: The user's question.
            k: Number of passages to ground the answer in.

        Returns:
            The grounded answer and its supporting sources (empty if refused).
        """
        with self._tracer.span("answer", query=query, k=k) as root:
            sources = self.search(query, k)
            contexts: Sequence[tuple[str, str]] = [
                (s.citation, s.text) for s in sources
            ]
            with self._tracer.span("generate", contexts=len(contexts)) as generation:
                text = generate_answer(query, contexts, self._provider)
                generation.update(output=text)
            # Deterministic output guardrails the prompt alone can't guarantee:
            # (1) a leaked system-prompt canary means the answer is compromised —
            # refuse outright; (2) otherwise strip any [n] citing past the retrieved
            # passages and refuse if the grounding was entirely fabricated.
            canary = screen_canary(text, SYSTEM_PROMPT_CANARY, refusal=REFUSAL)
            if canary.leaked:
                text, refused, stripped = canary.answer, True, 0
            else:
                validation = validate_citations(text, len(sources), refusal=REFUSAL)
                text = validation.answer
                refused = validation.refused or text.strip().startswith(REFUSAL[:20])
                stripped = len(validation.invalid_citations)
            root.update(
                metadata={
                    "refused": refused,
                    "sources": len(sources),
                    "stripped_citations": stripped,
                    "canary_leaked": canary.leaked,
                }
            )
            return AnswerResponse(
                answer=text, refused=refused, sources=[] if refused else sources
            )
