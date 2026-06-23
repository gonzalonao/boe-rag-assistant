"""Hybrid retrieval via Reciprocal Rank Fusion.

Combines the dense (semantic) and BM25 (lexical) retrievers, which fail in
different ways: dense matches paraphrases but misses exact legal references;
BM25 nails exact terms but misses synonyms. Reciprocal Rank Fusion (RRF) merges
their *rankings* — not their scores — so no score normalisation is needed and a
chunk ranked highly by either leg surfaces in the fused result.

RRF score for a chunk is ``sum(1 / (k_rrf + rank))`` over the rankings it
appears in (Cormack et al., 2009). The constant ``k_rrf`` damps the influence of
top ranks; 60 is the standard default.
"""

from __future__ import annotations

from collections.abc import Sequence

from boe_rag.eval.retriever import Searcher
from boe_rag.eval.sparse import BM25Index

#: Standard RRF damping constant (Cormack et al., 2009).
DEFAULT_K_RRF = 60
#: Candidates pulled from each leg before fusion.
DEFAULT_CANDIDATES = 50


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k_rrf: int = DEFAULT_K_RRF
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one by Reciprocal Rank Fusion.

    Args:
        rankings: Each inner sequence is chunk ids in descending rank order.
        k_rrf: RRF damping constant; larger values flatten rank influence.

    Returns:
        ``(chunk_id, fused_score)`` pairs sorted by descending fused score.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion.

    Both legs are indexed by their owners before being handed in; the dense leg
    is any :class:`~boe_rag.eval.retriever.Searcher`, so the in-memory NumPy
    index and a Qdrant-backed one are interchangeable here without touching
    fusion.

    Args:
        dense: The dense (embedding) retriever — any indexed ``Searcher``.
        sparse: The BM25 lexical retriever.
        k_rrf: RRF damping constant.
        candidates: Candidates pulled from each leg before fusion; capped to at
            least the requested ``k`` at search time.
    """

    def __init__(
        self,
        dense: Searcher,
        sparse: BM25Index,
        k_rrf: int = DEFAULT_K_RRF,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        """Bind the two retrieval legs and fusion hyperparameters."""
        self._dense = dense
        self._sparse = sparse
        self._k_rrf = k_rrf
        self._candidates = candidates

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Retrieve from both legs and return the top-k fused results.

        Args:
            query: The search query.
            k: Maximum number of fused results to return.

        Returns:
            Up to ``k`` ``(chunk_id, rrf_score)`` pairs, highest score first.
        """
        pool = max(self._candidates, k)
        dense_ids = [cid for cid, _ in self._dense.search(query, pool)]
        sparse_ids = [cid for cid, _ in self._sparse.search(query, pool)]
        return reciprocal_rank_fusion([dense_ids, sparse_ids], self._k_rrf)[:k]
