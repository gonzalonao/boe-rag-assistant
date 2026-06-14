"""Tests for grounded generation, the LLM judge, and the e2e runner.

A routing fake provider stands in for a real LLM: it inspects the system prompt
to decide whether it is being asked to generate an answer or to judge, so the
whole end-to-end flow is covered without any API key.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from boe_rag.eval.answerer import REFUSAL, generate_answer
from boe_rag.eval.dataset import EvalExample
from boe_rag.eval.e2e import run_e2e_eval
from boe_rag.eval.judge import judge_correctness, judge_faithfulness
from boe_rag.llm.base import ChatMessage, LLMError

_CONTEXTS = [
    (
        "Ley 19/2023, Artículo 1",
        "El Instituto Vasco de Finanzas tiene su sede en Bilbao.",
    ),
    (
        "Orden AUC/76/2024, Artículo 1",
        "Se eleva la oficina de Triesen a Consulado Honorario.",
    ),
]


class _CapturingProvider:
    """Records the last messages and returns a fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last: list[ChatMessage] = []

    @property
    def name(self) -> str:
        return "capturing"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.last = list(messages)
        return self.reply


class _RoutingProvider:
    """Returns an answer or a judge JSON depending on the system prompt."""

    def __init__(self, *, answer: str, faithfulness: float, correctness: float) -> None:
        self._answer = answer
        self._faith = faithfulness
        self._correct = correctness

    @property
    def name(self) -> str:
        return "routing"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        system = messages[0].content
        if "fidelidad" in system or "fundamentada" in system:
            return f'{{"score": {self._faith}, "reasoning": "ok"}}'
        if "corrección" in system or "REFERENCIA" in system:
            return f'{{"score": {self._correct}, "reasoning": "ok"}}'
        return self._answer


class _StubRetriever:
    """Returns a fixed ranking regardless of the query."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return [(cid, 1.0) for cid in self._ids[:k]]


def test_generate_answer_includes_contexts_and_question() -> None:
    """The prompt contains the numbered contexts and the question."""
    provider = _CapturingProvider("Tiene su sede en Bilbao [1].")
    answer = generate_answer("¿Dónde está la sede?", _CONTEXTS, provider)
    assert answer == "Tiene su sede en Bilbao [1]."
    user = provider.last[-1].content
    assert "[1] (Ley 19/2023, Artículo 1)" in user
    assert "¿Dónde está la sede?" in user
    assert REFUSAL in provider.last[0].content


def test_judge_faithfulness_parses_score() -> None:
    """A clean JSON judge reply is parsed into a score."""
    provider = _CapturingProvider('{"score": 0.9, "reasoning": "apoyado"}')
    result = judge_faithfulness("respuesta", _CONTEXTS, provider)
    assert result.metric == "faithfulness"
    assert result.score == pytest.approx(0.9)


def test_judge_tolerates_code_fences_and_clamps() -> None:
    """JSON wrapped in prose/code fences parses, and scores are clamped."""
    provider = _CapturingProvider('Aquí tienes:\n```json\n{"score": 1.4}\n```')
    result = judge_correctness("p", "a", "ref", provider)
    assert result.score == 1.0


def test_judge_rejects_unparseable_reply() -> None:
    """A reply with no JSON object raises an LLMError."""
    provider = _CapturingProvider("no puedo evaluar esto")
    with pytest.raises(LLMError):
        judge_faithfulness("respuesta", _CONTEXTS, provider)


def test_run_e2e_eval_aggregates_scores() -> None:
    """The e2e runner aggregates faithfulness, correctness, and refusals."""
    lookup = {"c1": _CONTEXTS[0], "c2": _CONTEXTS[1]}
    examples = [
        EvalExample(
            example_id="e1",
            question="¿Dónde está la sede?",
            relevant_chunk_ids=("c1",),
            answer="En Bilbao.",
        ),
    ]
    provider = _RoutingProvider(
        answer="La sede está en Bilbao [1].", faithfulness=0.8, correctness=1.0
    )
    metrics, results = run_e2e_eval(
        _StubRetriever(["c1", "c2"]), lookup, examples, provider, k=2
    )
    assert metrics.num_queries == 1
    assert metrics.mean_faithfulness == pytest.approx(0.8)
    assert metrics.mean_correctness == pytest.approx(1.0)
    assert metrics.refusal_rate == 0.0
    assert results[0].answer.startswith("La sede")


def test_run_e2e_eval_detects_refusal() -> None:
    """An answer equal to the refusal string is counted as a refusal."""
    lookup = {"c1": _CONTEXTS[0]}
    examples = [
        EvalExample(
            example_id="e1",
            question="¿Algo no presente?",
            relevant_chunk_ids=("c1",),
            answer="irrelevante",
        ),
    ]
    provider = _RoutingProvider(answer=REFUSAL, faithfulness=1.0, correctness=0.0)
    metrics, _ = run_e2e_eval(_StubRetriever(["c1"]), lookup, examples, provider, k=1)
    assert metrics.refusal_rate == 1.0
