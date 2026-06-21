"""Tests for the leakage-free document-level eval-set split."""

from __future__ import annotations

import pytest

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.split import document_id, split_by_document


def _example(example_id: str, *chunk_ids: str) -> EvalExample:
    """Build an example with a throwaway question and the given positives."""
    return EvalExample(
        example_id=example_id,
        question=f"question {example_id}",
        relevant_chunk_ids=chunk_ids,
    )


def _docs_of(examples: list[EvalExample]) -> set[str]:
    """Collect every source-document id referenced by ``examples``."""
    return {document_id(c) for ex in examples for c in ex.relevant_chunk_ids}


def _many(n: int) -> list[EvalExample]:
    """``n`` single-chunk examples, each from its own document."""
    return [_example(f"e{i}", f"BOE-A-2024-{i:04d}::0001") for i in range(n)]


def test_document_id_strips_section_suffix() -> None:
    """The document id is the chunk id up to the ``::`` separator."""
    assert document_id("BOE-A-2024-714::0004") == "BOE-A-2024-714"


def test_document_id_without_separator_is_identity() -> None:
    """A chunk id with no separator is its own document id."""
    assert document_id("BOE-A-2024-714") == "BOE-A-2024-714"


def test_split_is_deterministic_for_a_seed() -> None:
    """The same seed reproduces the exact same partition."""
    examples = _many(100)
    a_train, a_test = split_by_document(examples, test_fraction=0.2, seed=7)
    b_train, b_test = split_by_document(examples, test_fraction=0.2, seed=7)
    assert [e.example_id for e in a_train] == [e.example_id for e in b_train]
    assert [e.example_id for e in a_test] == [e.example_id for e in b_test]


def test_split_partitions_every_example_once() -> None:
    """Train and test together cover all examples with no overlap."""
    examples = _many(100)
    train, test = split_by_document(examples, test_fraction=0.3, seed=1)
    ids = {e.example_id for e in train} | {e.example_id for e in test}
    assert ids == {e.example_id for e in examples}
    assert len(train) + len(test) == len(examples)


def test_test_fraction_is_approximately_respected() -> None:
    """With one document per example the test share is close to the target."""
    examples = _many(200)
    _, test = split_by_document(examples, test_fraction=0.25, seed=3)
    assert abs(len(test) - 50) <= 1


def test_source_documents_are_disjoint_across_splits() -> None:
    """No source document appears in both the train and test splits."""
    examples = _many(120)
    train, test = split_by_document(examples, test_fraction=0.2, seed=11)
    assert _docs_of(train).isdisjoint(_docs_of(test))


def test_examples_sharing_a_document_stay_together() -> None:
    """Two questions on the same document are never split apart."""
    shared = "BOE-A-2024-0001::0002"
    paired_a = _example("pa", "BOE-A-2024-0001::0001")
    paired_b = _example("pb", shared)  # same document as paired_a
    examples = [paired_a, paired_b, *_many(40)]
    train, test = split_by_document(examples, test_fraction=0.3, seed=5)
    on_test = {"pa", "pb"} <= {e.example_id for e in test}
    on_train = {"pa", "pb"} <= {e.example_id for e in train}
    assert on_test or on_train


def test_multi_document_example_links_its_documents() -> None:
    """An example citing two documents keeps both on the same side."""
    cross = _example("x", "BOE-A-2024-0001::0001", "BOE-A-2024-0002::0001")
    examples = [cross, *_many(30)]
    train, test = split_by_document(examples, test_fraction=0.3, seed=9)
    assert _docs_of(train).isdisjoint(_docs_of(test))


def test_rejects_out_of_range_fraction() -> None:
    """A test fraction outside (0, 1) is rejected."""
    examples = _many(10)
    with pytest.raises(ValueError, match="test_fraction"):
        split_by_document(examples, test_fraction=0.0, seed=1)
    with pytest.raises(ValueError, match="test_fraction"):
        split_by_document(examples, test_fraction=1.0, seed=1)


def test_rejects_empty_input() -> None:
    """Splitting an empty list is an error."""
    with pytest.raises(ValueError, match="empty"):
        split_by_document([], test_fraction=0.2, seed=1)
