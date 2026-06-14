"""Two-stage retrieval: re-score a candidate pool with a reranker.

A bi-encoder (dense) retriever embeds the query and each passage independently,
so it can only approximate how well they actually match. A cross-encoder reads a
query and a candidate passage *together* and scores their relevance directly —
much sharper, but far too slow to run over the whole corpus. The standard remedy
is a two-stage pipeline: a cheap first-stage retriever proposes a small candidate
pool, then the reranker reorders just that pool.

This module is the pipeline glue. The :class:`Reranker` protocol keeps the heavy
model out of the logic (and out of CI); the concrete cross-encoder lives in
:mod:`boe_rag.eval.cross_encoder`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from boe_rag.eval.retriever import Searcher

#: Default number of first-stage candidates fed to the reranker.
DEFAULT_RERANK_POOL = 30


class Reranker(Protocol):
    """Re-scores ``(chunk_id, text)`` candidates against a query."""

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Return the candidates as ``(chunk_id, score)``, best first."""
        ...


class RerankingRetriever:
    """Wraps a first-stage retriever with a reranking second stage.

    Args:
        base: The first-stage retriever proposing candidates.
        reranker: The reranker that reorders the candidate pool.
        id_to_text: Maps a chunk id to its text (the reranker needs the passage).
        pool: Number of first-stage candidates to rerank; raised to the
            requested ``k`` at search time if smaller.
    """

    def __init__(
        self,
        base: Searcher,
        reranker: Reranker,
        id_to_text: Mapping[str, str],
        pool: int = DEFAULT_RERANK_POOL,
    ) -> None:
        """Bind the first stage, the reranker, and the id→text lookup."""
        self._base = base
        self._reranker = reranker
        self._id_to_text = id_to_text
        self._pool = pool

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Retrieve a pool, rerank it, and return the top-k.

        Args:
            query: The search query.
            k: Maximum number of reranked results to return.

        Returns:
            Up to ``k`` ``(chunk_id, rerank_score)`` pairs, highest score first.
        """
        pool = max(self._pool, k)
        candidates = self._base.search(query, pool)
        pairs = [
            (cid, self._id_to_text[cid])
            for cid, _ in candidates
            if cid in self._id_to_text
        ]
        if not pairs:
            return []
        return self._reranker.rerank(query, pairs)[:k]
