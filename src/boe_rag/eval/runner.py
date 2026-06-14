"""Run a retrieval evaluation and aggregate metrics.

Glue between the corpus, the golden eval set, a retriever, and the metric
functions: index the corpus, retrieve top-k for each question, and score the
rankings against the ground-truth relevant chunk ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.metrics import RetrievalMetrics, evaluate_retrieval
from boe_rag.eval.retriever import DenseRetriever, Embedder


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """Per-question retrieval outcome, retained for error analysis.

    Attributes:
        example_id: The evaluated example's id.
        retrieved_ids: Chunk ids returned, in rank order.
        relevant_ids: Ground-truth relevant chunk ids.
        first_relevant_rank: 1-based rank of the first hit, or ``None`` if missed.
    """

    example_id: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    first_relevant_rank: int | None


def _first_relevant_rank(
    retrieved: Sequence[str], relevant: frozenset[str]
) -> int | None:
    """Return the 1-based rank of the first relevant id, or ``None``."""
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return rank
    return None


def run_retrieval_eval(
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    examples: Sequence[EvalExample],
    embedder: Embedder,
    k: int = 10,
    retrieve_n: int | None = None,
) -> tuple[RetrievalMetrics, list[ExampleResult]]:
    """Index the corpus and evaluate the retriever over the eval set.

    Args:
        chunk_ids: Corpus chunk ids, aligned with ``texts``.
        texts: Corpus chunk texts.
        examples: Golden eval examples.
        embedder: Embedder for indexing and querying.
        k: Cut-off rank for the reported @k metrics.
        retrieve_n: How many candidates to retrieve per query; defaults to ``k``.
            Set larger than ``k`` to also measure deeper recall.

    Returns:
        The aggregated metrics and the per-example results.
    """
    depth = retrieve_n or k
    retriever = DenseRetriever(embedder)
    retriever.index(chunk_ids, texts)

    scored: list[tuple[Sequence[str], frozenset[str]]] = []
    results: list[ExampleResult] = []
    for example in examples:
        retrieved = [cid for cid, _ in retriever.search(example.question, depth)]
        relevant = frozenset(example.relevant_chunk_ids)
        scored.append((retrieved, relevant))
        results.append(
            ExampleResult(
                example_id=example.example_id,
                retrieved_ids=tuple(retrieved),
                relevant_ids=example.relevant_chunk_ids,
                first_relevant_rank=_first_relevant_rank(retrieved, relevant),
            )
        )
    metrics = evaluate_retrieval(scored, k=k)
    return metrics, results
