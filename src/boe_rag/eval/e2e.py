"""End-to-end RAG evaluation: retrieve, generate, then judge.

Runs the full baseline pipeline for each golden question and scores the
generated answer with the LLM-as-judge. Produces the end-to-end "before"
numbers (faithfulness, correctness, refusal rate) that complement the
retrieval metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from boe_rag.eval.answerer import REFUSAL, generate_answer
from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.judge import judge_correctness, judge_faithfulness
from boe_rag.eval.security import citation_indices
from boe_rag.llm.base import LLMProvider


class SupportsSearch(Protocol):
    """Anything that can return ranked chunk ids for a query."""

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` pairs, best first."""
        ...


@dataclass(frozen=True, slots=True)
class E2EExampleResult:
    """Per-question end-to-end outcome.

    Attributes:
        example_id: The evaluated example's id.
        answer: The generated answer.
        faithfulness: Faithfulness score in ``[0, 1]``.
        correctness: Correctness score in ``[0, 1]``.
        refused: Whether the model emitted the refusal string.
        cited: Whether the answer carried at least one valid citation (an index in
            ``1..num_passages``). Used to measure the cite-or-refuse guardrail's
            false-positive surface: a non-refused answer that is not cited is one the
            ``service.citation`` invariant would convert to a refusal.
    """

    example_id: str
    answer: str
    faithfulness: float
    correctness: float
    refused: bool
    cited: bool


@dataclass(frozen=True, slots=True)
class E2EMetrics:
    """Aggregated end-to-end metrics.

    Attributes:
        num_queries: Number of questions evaluated.
        mean_faithfulness: Mean faithfulness over all questions.
        mean_correctness: Mean correctness over all questions.
        refusal_rate: Fraction of questions the model refused to answer.
        uncited_answer_rate: Fraction of *answered* (non-refused) questions whose
            answer carried no valid citation. This is the cite-or-refuse guardrail's
            false-positive rate on legitimate traffic: each such answer would be
            converted to a refusal by ``service.citation.validate_citations``. ``0.0``
            when every question was refused.
    """

    num_queries: int
    mean_faithfulness: float
    mean_correctness: float
    refusal_rate: float
    uncited_answer_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict for serialisation."""
        return asdict(self)


def run_e2e_eval(
    retriever: SupportsSearch,
    chunk_lookup: Mapping[str, tuple[str, str]],
    examples: Sequence[EvalExample],
    provider: LLMProvider,
    k: int = 5,
) -> tuple[E2EMetrics, list[E2EExampleResult]]:
    """Run retrieve → generate → judge for every example and aggregate.

    Args:
        retriever: An indexed retriever.
        chunk_lookup: Maps a chunk id to its ``(citation, text)``.
        examples: Golden eval examples (their ``answer`` is the reference).
        provider: LLM provider used for both generation and judging.
        k: Number of passages to retrieve and pass to the generator.

    Returns:
        The aggregated metrics and the per-example results.
    """
    results: list[E2EExampleResult] = []
    for example in examples:
        retrieved = [cid for cid, _ in retriever.search(example.question, k)]
        contexts = [chunk_lookup[cid] for cid in retrieved if cid in chunk_lookup]
        answer = generate_answer(example.question, contexts, provider)
        refused = answer.strip().startswith(REFUSAL[:20])
        cited = any(1 <= i <= len(contexts) for i in citation_indices(answer))
        faithfulness = judge_faithfulness(answer, contexts, provider)
        correctness = judge_correctness(
            example.question, answer, example.answer or "", provider
        )
        results.append(
            E2EExampleResult(
                example_id=example.example_id,
                answer=answer,
                faithfulness=faithfulness.score,
                correctness=correctness.score,
                refused=refused,
                cited=cited,
            )
        )
    return _aggregate(results), results


def _aggregate(results: Sequence[E2EExampleResult]) -> E2EMetrics:
    """Average per-example results into :class:`E2EMetrics`."""
    n = len(results)
    if n == 0:
        return E2EMetrics(0, 0.0, 0.0, 0.0, 0.0)
    answered = [r for r in results if not r.refused]
    uncited = sum(1 for r in answered if not r.cited)
    return E2EMetrics(
        num_queries=n,
        mean_faithfulness=sum(r.faithfulness for r in results) / n,
        mean_correctness=sum(r.correctness for r in results) / n,
        refusal_rate=sum(1 for r in results if r.refused) / n,
        uncited_answer_rate=uncited / len(answered) if answered else 0.0,
    )
