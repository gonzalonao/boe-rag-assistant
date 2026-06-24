"""Tests for byte-identical text equivalence and its effect on scoring."""

from __future__ import annotations

import pytest

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.equivalence import build_text_equivalence
from boe_rag.eval.runner import evaluate_searcher


class _FixedSearcher:
    """Returns a fixed ranked list of (chunk_id, score), ignoring the query."""

    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self._ranked = ranked

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._ranked[:k]


def test_singletons_are_their_own_representative() -> None:
    """Unique texts map to themselves; nothing is folded."""
    eq = build_text_equivalence(["a", "b", "c"], ["one", "two", "three"])
    assert eq.num_redundant == 0
    assert eq.representative("a") == "a"
    assert eq.representative("absent") == "absent"


def test_identical_texts_share_canonical_representative() -> None:
    """Byte-identical chunks fold onto the lexicographically smallest id."""
    eq = build_text_equivalence(
        ["d2::1", "d1::4", "d3::0"], ["same clause", "same clause", "same clause"]
    )
    assert eq.num_redundant == 2  # two non-canonical duplicates
    rep = eq.representative("d2::1")
    assert rep == "d1::4"  # min() of the three ids
    assert eq.representative("d1::4") == "d1::4"
    assert eq.representative("d3::0") == "d1::4"


def test_canonical_set_collapses_duplicates() -> None:
    """A relevant set of duplicates collapses to a single class."""
    eq = build_text_equivalence(["x", "y"], ["dup", "dup"])
    assert eq.canonical_set(["x", "y"]) == frozenset({"x"})


def test_canonical_sequence_dedups_preserving_order() -> None:
    """A ranking of duplicates collapses to the first rank of each class."""
    eq = build_text_equivalence(
        ["b", "a", "c"], ["dup", "dup", "unique"]
    )  # a,b -> a ; c -> c
    assert eq.canonical_sequence(["b", "c", "a"]) == ["a", "c"]


def test_only_exact_matches_fold() -> None:
    """Texts that differ by a single character are not folded."""
    eq = build_text_equivalence(["a", "b"], ["clause.", "clause,"])
    assert eq.num_redundant == 0


def test_build_rejects_length_mismatch() -> None:
    """Mismatched chunk_ids/texts lengths are rejected."""
    with pytest.raises(ValueError, match="same length"):
        build_text_equivalence(["a"], ["x", "y"])


def test_equivalence_turns_a_tie_miss_into_a_hit() -> None:
    """The q003 scenario: a gold twin outside top-k, an identical twin inside it.

    Three byte-identical clauses live in different documents. The retriever ranks
    a non-gold twin first and the gold-labelled copy far down. Without
    equivalence, recall@2 is 0 (the labelled copy is missed); with equivalence,
    retrieving an interchangeable identical copy counts as a hit.
    """
    clause = "impuestos no incluidos"
    chunk_ids = ["d2::4", "d1::4", "d3::4", "other"]
    texts = [clause, clause, clause, "x"]
    eq = build_text_equivalence(chunk_ids, texts)

    # Ranking surfaces a non-gold identical twin (d1::4) at rank 1; the gold copy
    # (d3::4) is at rank 4, outside k=2.
    searcher = _FixedSearcher(
        [("d1::4", 0.9), ("other", 0.8), ("d2::4", 0.7), ("d3::4", 0.7)]
    )
    example = EvalExample(
        example_id="q003",
        question="¿qué impuestos no se incluyen?",
        relevant_chunk_ids=("d3::4",),
    )

    raw, _ = evaluate_searcher(searcher, [example], k=2)
    assert raw.recall_at_k == 0.0

    fair, results = evaluate_searcher(searcher, [example], k=2, equivalence=eq)
    assert fair.recall_at_k == 1.0
    assert results[0].first_relevant_rank == 1
