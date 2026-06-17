"""Tests for the eval dataset schema, persistence, and the seed set."""

from __future__ import annotations

from pathlib import Path

import pytest

from boe_rag.eval.dataset import EvalExample, load_evalset, save_evalset

_SEED_EVALSET = Path(__file__).parents[1] / "eval_data" / "seed_evalset.jsonl"


def test_eval_example_requires_relevant_chunks() -> None:
    """An example must reference at least one relevant chunk."""
    with pytest.raises(ValueError):
        EvalExample(example_id="q", question="?", relevant_chunk_ids=())


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Examples survive a save/load cycle unchanged."""
    examples = [
        EvalExample(
            example_id="q1",
            question="¿Cuál es la sede?",
            relevant_chunk_ids=("BOE-A-2024-1::0000",),
            answer="Bilbao.",
            category="test",
            difficulty="easy",
        ),
        EvalExample(
            example_id="q2",
            question="¿Y el importe?",
            relevant_chunk_ids=("BOE-A-2024-2::0001", "BOE-A-2024-2::0002"),
        ),
    ]
    path = tmp_path / "sub" / "evalset.jsonl"
    assert save_evalset(examples, path) == 2
    assert load_evalset(path) == examples


def test_load_evalset_missing_file(tmp_path: Path) -> None:
    """Loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_evalset(tmp_path / "nope.jsonl")


def test_seed_evalset_is_valid() -> None:
    """The committed seed eval set loads and is well-formed."""
    examples = load_evalset(_SEED_EVALSET)
    assert len(examples) >= 15
    ids = [e.example_id for e in examples]
    assert len(set(ids)) == len(ids), "example ids must be unique"
    for example in examples:
        assert example.question.endswith("?")
        assert all(cid.startswith("BOE-") for cid in example.relevant_chunk_ids)
        assert example.answer
