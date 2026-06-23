"""Tests for contrastive training-pair mining.

A fake searcher returns a fixed ranking so the negative-selection rules
(exclude positives, skip unknown ids, respect the count) are covered without
BM25 or the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.mine_pairs import (
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    TrainingPair,
    build_columnar_dataset,
    build_training_pairs,
    load_pairs,
    mine_negative_texts,
    save_pairs,
)

_LOOKUP = {
    "c1": "texto uno",
    "c2": "texto dos",
    "c3": "texto tres",
    "c4": "texto cuatro",
}


class _FakeSearcher:
    """Returns a fixed id ranking, ignoring the query and k beyond truncation."""

    def __init__(self, ranking: Sequence[str]) -> None:
        """Store the ranking the fake will return for any query."""
        self._ranking = list(ranking)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the canned ranking with descending placeholder scores."""
        return [(cid, 1.0 - i * 0.1) for i, cid in enumerate(self._ranking[:k])]


def test_mine_negatives_excludes_positive_and_respects_count() -> None:
    """The positive is skipped and only num_negatives texts are returned."""
    searcher = _FakeSearcher(["c1", "c2", "c3", "c4"])
    negatives = mine_negative_texts(
        searcher,
        "q",
        frozenset({"c1"}),
        _LOOKUP,
        num_negatives=2,
        pool=10,
    )
    assert negatives == ("texto dos", "texto tres")


def test_mine_negatives_skips_ids_missing_from_lookup() -> None:
    """A retrieved id absent from the corpus map is ignored, not crashed on."""
    searcher = _FakeSearcher(["c1", "ghost", "c3"])
    negatives = mine_negative_texts(
        searcher,
        "q",
        frozenset({"c1"}),
        _LOOKUP,
        num_negatives=5,
        pool=10,
    )
    assert negatives == ("texto tres",)


def test_build_pairs_one_per_relevant_chunk_in_lookup() -> None:
    """Each relevant chunk present in the corpus yields one anchor/positive pair."""
    examples = [
        EvalExample(
            example_id="e1",
            question="pregunta uno",
            relevant_chunk_ids=("c1",),
        ),
        EvalExample(
            example_id="e2",
            question="pregunta dos",
            relevant_chunk_ids=("c2", "missing"),
        ),
    ]
    searcher = _FakeSearcher(["c3", "c4", "c1", "c2"])
    pairs = build_training_pairs(examples, _LOOKUP, searcher, num_negatives=2, pool=10)

    assert [(p.query, p.positive) for p in pairs] == [
        ("pregunta uno", "texto uno"),
        ("pregunta dos", "texto dos"),
    ]
    # Negatives exclude the example's own positives and come from the ranking.
    assert pairs[0].negatives == ("texto tres", "texto cuatro")


def test_build_pairs_skips_example_with_no_known_positive() -> None:
    """An example whose relevant chunks are all unknown contributes nothing."""
    examples = [
        EvalExample(
            example_id="e1",
            question="huérfana",
            relevant_chunk_ids=("ghost1", "ghost2"),
        )
    ]
    pairs = build_training_pairs(
        examples, _LOOKUP, _FakeSearcher(["c1", "c2"]), num_negatives=1
    )
    assert pairs == []


def test_build_pairs_without_negatives_when_count_zero() -> None:
    """num_negatives=0 yields positives only and never queries the searcher."""
    examples = [EvalExample(example_id="e1", question="q", relevant_chunk_ids=("c1",))]

    class _Boom:
        def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
            raise AssertionError("searcher must not be called when num_negatives=0")

    pairs = build_training_pairs(examples, _LOOKUP, _Boom(), num_negatives=0)
    assert pairs == [TrainingPair("q", "texto uno", ())]


def test_columnar_dataset_prefixes_and_drops_short_rows() -> None:
    """Texts get E5 prefixes and rows lacking enough negatives are dropped."""
    pairs = [
        TrainingPair("q1", "p1", ("n1a", "n1b")),
        TrainingPair("q2", "p2", ("n2a",)),  # too few negatives -> dropped
    ]
    columns = build_columnar_dataset(pairs, num_negatives=2)

    assert columns["anchor"] == [f"{E5_QUERY_PREFIX}q1"]
    assert columns["positive"] == [f"{E5_PASSAGE_PREFIX}p1"]
    assert columns["negative_1"] == [f"{E5_PASSAGE_PREFIX}n1a"]
    assert columns["negative_2"] == [f"{E5_PASSAGE_PREFIX}n1b"]


def test_columnar_dataset_without_negative_columns() -> None:
    """num_negatives=0 emits only anchor/positive columns, keeping every pair."""
    pairs = [TrainingPair("q", "p", ())]
    columns = build_columnar_dataset(pairs, num_negatives=0)
    assert set(columns) == {"anchor", "positive"}
    assert columns["anchor"] == [f"{E5_QUERY_PREFIX}q"]


def test_pairs_round_trip_through_jsonl(tmp_path: Path) -> None:
    """Saved pairs load back identical, including unicode and empty negatives."""
    pairs = [
        TrainingPair("¿qué es?", "respuesta ñ", ("neg á", "neg é")),
        TrainingPair("sin negativos", "positivo", ()),
    ]
    out = tmp_path / "pairs.jsonl"

    assert save_pairs(pairs, out) == 2
    assert load_pairs(out) == pairs
