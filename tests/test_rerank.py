"""Tests for the reranking retriever, using a fake reranker (no torch)."""

from __future__ import annotations

from collections.abc import Sequence

from boe_rag.eval.rerank import RerankingRetriever


class _FakeBase:
    """First-stage retriever returning a fixed ranking."""

    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        # Descending pseudo-scores in the given order.
        return [
            (cid, float(len(self._ranking) - i)) for i, cid in enumerate(self._ranking)
        ][:k]


class _PreferenceReranker:
    """Reranker that scores candidates by a fixed id→score preference."""

    def __init__(self, preference: dict[str, float]) -> None:
        self._preference = preference

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        scored = [(cid, self._preference.get(cid, 0.0)) for cid, _ in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


_ID_TO_TEXT = {"c1": "texto uno", "c2": "texto dos", "c3": "texto tres"}


def test_rerank_reorders_first_stage_ranking() -> None:
    """The reranker can promote a candidate the base ranked last."""
    base = _FakeBase(["c1", "c2", "c3"])
    reranker = _PreferenceReranker({"c3": 9.0, "c1": 1.0, "c2": 0.5})
    retriever = RerankingRetriever(base, reranker, _ID_TO_TEXT, pool=3)
    results = retriever.search("q", k=3)
    assert [cid for cid, _ in results] == ["c3", "c1", "c2"]


def test_rerank_respects_k() -> None:
    """Only the top-k reranked results are returned."""
    base = _FakeBase(["c1", "c2", "c3"])
    reranker = _PreferenceReranker({"c2": 5.0, "c1": 4.0, "c3": 3.0})
    retriever = RerankingRetriever(base, reranker, _ID_TO_TEXT, pool=3)
    assert len(retriever.search("q", k=1)) == 1


def test_rerank_pool_at_least_k() -> None:
    """The pool is widened to k so a small pool never starves the top-k."""
    base = _FakeBase(["c1", "c2", "c3"])
    reranker = _PreferenceReranker({"c1": 1.0, "c2": 2.0, "c3": 3.0})
    retriever = RerankingRetriever(base, reranker, _ID_TO_TEXT, pool=1)
    results = retriever.search("q", k=3)
    assert [cid for cid, _ in results] == ["c3", "c2", "c1"]


def test_rerank_skips_ids_without_text() -> None:
    """Candidates missing from the lookup are dropped before reranking."""
    base = _FakeBase(["c1", "missing", "c2"])
    reranker = _PreferenceReranker({"c1": 1.0, "c2": 2.0})
    retriever = RerankingRetriever(base, reranker, _ID_TO_TEXT, pool=3)
    results = retriever.search("q", k=3)
    assert [cid for cid, _ in results] == ["c2", "c1"]
