"""Mine contrastive training pairs for fine-tuning the embedding model.

Fine-tuning a retriever with a contrastive loss needs, per example, an *anchor*
(the question), a *positive* (the chunk that answers it), and ideally a few
*hard negatives* — passages that look relevant but are not. The silver eval set
already supplies (question → relevant chunk) positives at scale; this module adds
the hard negatives by asking a lexical retriever for the question's top matches
and keeping the near-misses that are **not** the gold chunk. Those are exactly
the confusable passages the dense model most needs to learn to push apart.

The logic here is pure and retriever-agnostic (it depends only on the
:class:`~boe_rag.eval.retriever.Searcher` protocol and a chunk-id → text map), so
it is unit-tested without the corpus, BM25, or the ``ml`` extra. Mined pairs hold
**raw** text; the E5 ``query:``/``passage:`` prefixes are an encoder detail applied
later, at training time, in ``scripts/finetune_embeddings.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.retriever import Searcher

#: Default number of hard negatives to mine per positive.
DEFAULT_NUM_NEGATIVES = 4
#: Default candidate pool pulled from the retriever before filtering positives.
DEFAULT_POOL = 20

#: E5 models are trained with these role prefixes; queries and passages must
#: carry them at both training and inference time for the embeddings to align.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True, slots=True)
class TrainingPair:
    """One contrastive training example.

    Attributes:
        query: The anchor question.
        positive: Text of the chunk that answers ``query``.
        negatives: Texts of hard-negative chunks (lexical near-misses that are
            not the positive); may be empty if none were available.
    """

    query: str
    positive: str
    negatives: tuple[str, ...]


def mine_negative_texts(
    searcher: Searcher,
    query: str,
    exclude_ids: frozenset[str],
    lookup: Mapping[str, str],
    *,
    num_negatives: int = DEFAULT_NUM_NEGATIVES,
    pool: int = DEFAULT_POOL,
) -> tuple[str, ...]:
    """Return texts of hard-negative chunks for ``query``.

    Retrieves the top ``pool`` candidates, drops any in ``exclude_ids`` (the
    gold positives) or absent from ``lookup``, and keeps the first
    ``num_negatives`` by retrieval rank.

    Args:
        searcher: Retriever used to surface confusable candidates (typically the
            lexical BM25 leg, whose near-misses are the hardest negatives).
        query: The anchor question.
        exclude_ids: Chunk ids that must never be used as negatives (positives).
        lookup: Chunk id → text map for the corpus.
        num_negatives: Maximum number of negatives to return.
        pool: Candidates to retrieve before filtering.

    Returns:
        Up to ``num_negatives`` hard-negative chunk texts, in rank order.
    """
    negatives: list[str] = []
    for chunk_id, _score in searcher.search(query, max(pool, num_negatives)):
        if chunk_id in exclude_ids:
            continue
        text = lookup.get(chunk_id)
        if text is None:
            continue
        negatives.append(text)
        if len(negatives) >= num_negatives:
            break
    return tuple(negatives)


def build_training_pairs(
    examples: Sequence[EvalExample],
    lookup: Mapping[str, str],
    searcher: Searcher,
    *,
    num_negatives: int = DEFAULT_NUM_NEGATIVES,
    pool: int = DEFAULT_POOL,
) -> list[TrainingPair]:
    """Build contrastive training pairs from an eval set and corpus.

    Each example contributes one pair per relevant chunk that is present in
    ``lookup``: the question as anchor, that chunk's text as positive, and hard
    negatives mined for the question (excluding *all* of the example's relevant
    chunks). Examples whose relevant chunks are all missing from ``lookup`` are
    skipped, so a stale eval set can never inject empty positives.

    Args:
        examples: Silver/gold examples supplying (question, relevant-chunk) pairs.
        lookup: Chunk id → text map for the corpus.
        searcher: Retriever used to mine hard negatives.
        num_negatives: Hard negatives to mine per positive.
        pool: Candidate pool pulled from the retriever per question.

    Returns:
        The mined training pairs, in example order.
    """
    exclude_all = num_negatives <= 0
    pairs: list[TrainingPair] = []
    for example in examples:
        relevant = frozenset(example.relevant_chunk_ids)
        for chunk_id in example.relevant_chunk_ids:
            positive = lookup.get(chunk_id)
            if positive is None:
                continue
            negatives = (
                ()
                if exclude_all
                else mine_negative_texts(
                    searcher,
                    example.question,
                    relevant,
                    lookup,
                    num_negatives=num_negatives,
                    pool=pool,
                )
            )
            pairs.append(
                TrainingPair(
                    query=example.question,
                    positive=positive,
                    negatives=negatives,
                )
            )
    return pairs


def build_columnar_dataset(
    pairs: Sequence[TrainingPair], num_negatives: int
) -> dict[str, list[str]]:
    """Lay mined pairs out as fixed-width, E5-prefixed training columns.

    The sentence-transformers trainer consumes a rectangular dataset, so every
    row must have the same columns: an ``anchor``, a ``positive``, and exactly
    ``num_negatives`` ``negative_i`` columns. Pairs with fewer mined negatives
    than ``num_negatives`` are dropped (they would leave ragged rows); with a
    healthy candidate pool this is rare. Every text gets its E5 role prefix here,
    once, so the trainer and inference encoder agree.

    Args:
        pairs: Mined training pairs.
        num_negatives: Exact number of negative columns to emit per row.

    Returns:
        A column-name → values mapping with keys ``anchor``, ``positive``, and
        ``negative_1`` … ``negative_{num_negatives}``. Empty lists if no pair
        has enough negatives.

    Raises:
        ValueError: If ``num_negatives`` is negative.
    """
    if num_negatives < 0:
        raise ValueError(f"num_negatives must be >= 0, got {num_negatives}")
    columns: dict[str, list[str]] = {"anchor": [], "positive": []}
    for i in range(1, num_negatives + 1):
        columns[f"negative_{i}"] = []
    for pair in pairs:
        if len(pair.negatives) < num_negatives:
            continue
        columns["anchor"].append(f"{E5_QUERY_PREFIX}{pair.query}")
        columns["positive"].append(f"{E5_PASSAGE_PREFIX}{pair.positive}")
        for i in range(1, num_negatives + 1):
            columns[f"negative_{i}"].append(
                f"{E5_PASSAGE_PREFIX}{pair.negatives[i - 1]}"
            )
    return columns


def save_pairs(pairs: Iterable[TrainingPair], path: Path) -> int:
    """Write training pairs to a JSONL file, creating parent dirs as needed.

    Args:
        pairs: The pairs to serialise.
        path: Destination ``.jsonl`` path.

    Returns:
        The number of pairs written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "query": pair.query,
                        "positive": pair.positive,
                        "negatives": list(pair.negatives),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
            count += 1
    return count


def load_pairs(path: Path) -> list[TrainingPair]:
    """Load training pairs written by :func:`save_pairs`.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        The parsed pairs in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Training pairs not found: {path}")
    pairs: list[TrainingPair] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            pairs.append(
                TrainingPair(
                    query=str(payload["query"]),
                    positive=str(payload["positive"]),
                    negatives=tuple(str(n) for n in payload["negatives"]),
                )
            )
    return pairs
