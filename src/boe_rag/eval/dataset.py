"""Golden evaluation dataset: schema and JSONL persistence.

An eval example is a natural-language question paired with the chunk id(s) that
contain its answer. This is the ground truth the retriever is scored against.
The set is stored as JSONL (one example per line) so it diffs cleanly in git
and streams without loading everything into memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvalExample(BaseModel):
    """A single golden question with its relevant chunk(s).

    Attributes:
        example_id: Stable identifier for the example.
        question: Natural-language question, as a user would phrase it.
        relevant_chunk_ids: Chunk ids that answer the question (at least one).
        answer: Optional reference answer, used by end-to-end (generation) evals.
        category: Optional topic/category tag for slicing metrics.
        difficulty: Optional difficulty tag (e.g. ``easy``/``medium``/``hard``).
    """

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...] = Field(min_length=1)
    answer: str | None = None
    category: str | None = None
    difficulty: str | None = None


def load_evalset(path: Path) -> list[EvalExample]:
    """Load a JSONL evaluation set.

    Args:
        path: Path to a ``.jsonl`` file, one :class:`EvalExample` per line.

    Returns:
        The parsed examples in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or a valid example.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Eval set not found: {path}")
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                examples.append(EvalExample.model_validate_json(line))
            except ValueError as err:
                raise ValueError(
                    f"{path}:{line_number}: invalid example: {err}"
                ) from err
    return examples


def save_evalset(examples: Iterable[EvalExample], path: Path) -> int:
    """Write examples to a JSONL file, creating parent directories as needed.

    Args:
        examples: The examples to serialise.
        path: Destination ``.jsonl`` path.

    Returns:
        The number of examples written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json())
            handle.write("\n")
            count += 1
    return count
