"""LLM-as-judge metrics for end-to-end answer quality.

Two complementary judgments, each a single rate-limited LLM call returning a
0-1 score with a short rationale:

* **faithfulness** — is the answer grounded only in the retrieved passages
  (no hallucination)?
* **correctness** — does the answer actually answer the question, matching the
  reference answer?

Scores are deliberately on a continuous 0-1 scale so improvements are visible
between pipeline versions. The judge runs at ``temperature=0`` for stability.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from boe_rag.eval.answerer import build_context_block
from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider

#: First JSON object embedded anywhere in a model reply.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_FAITHFULNESS_SYSTEM = (
    "Eres un evaluador estricto. Determina en qué medida la RESPUESTA está "
    "fundamentada ÚNICAMENTE en los FRAGMENTOS proporcionados. Una respuesta "
    "totalmente apoyada por los fragmentos puntúa 1; una que añade información "
    "no presente o la contradice puntúa cerca de 0. Devuelve SOLO un objeto "
    'JSON con las claves "score" (número entre 0 y 1) y "reasoning" (cadena '
    "breve en español)."
)

_CORRECTNESS_SYSTEM = (
    "Eres un evaluador. Compara la RESPUESTA con la RESPUESTA DE REFERENCIA "
    "para la PREGUNTA dada. Puntúa 1 si la respuesta es correcta y completa, "
    "valores intermedios si es parcial, y cerca de 0 si es incorrecta o evade "
    'la pregunta. Devuelve SOLO un objeto JSON con las claves "score" (número '
    'entre 0 y 1) y "reasoning" (cadena breve en español).'
)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """The outcome of a single judgment.

    Attributes:
        metric: The judged metric (``faithfulness`` or ``correctness``).
        score: Score in ``[0, 1]``.
        reasoning: The judge's short rationale.
    """

    metric: str
    score: float
    reasoning: str


def _parse_judgment(metric: str, raw: str) -> JudgeResult:
    """Parse a judge reply into a :class:`JudgeResult`.

    Args:
        metric: The metric name to tag the result with.
        raw: The model's raw reply (may include prose or code fences).

    Returns:
        The parsed result, with the score clamped to ``[0, 1]``.

    Raises:
        LLMError: If no score can be parsed from the reply.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise LLMError(f"judge reply has no JSON object: {raw!r}")
    try:
        payload = json.loads(match.group(0))
        score = float(payload["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
        raise LLMError(f"unparseable judge reply: {raw!r}") from err
    reasoning = str(payload.get("reasoning", "")).strip()
    return JudgeResult(
        metric=metric, score=max(0.0, min(1.0, score)), reasoning=reasoning
    )


def judge_faithfulness(
    answer: str,
    contexts: Sequence[tuple[str, str]],
    provider: LLMProvider,
) -> JudgeResult:
    """Judge how well an answer is grounded in the retrieved passages.

    Args:
        answer: The generated answer.
        contexts: ``(citation, text)`` pairs shown to the generator.
        provider: The LLM provider acting as judge.

    Returns:
        The faithfulness judgment.
    """
    user = (
        f"FRAGMENTOS:\n{build_context_block(contexts)}\n\n"
        f"RESPUESTA:\n{answer}\n\nEvalúa la fidelidad."
    )
    reply = provider.complete(
        [
            ChatMessage(role="system", content=_FAITHFULNESS_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _parse_judgment("faithfulness", reply)


def judge_correctness(
    question: str,
    answer: str,
    reference: str,
    provider: LLMProvider,
) -> JudgeResult:
    """Judge whether an answer correctly answers the question.

    Args:
        question: The question that was asked.
        answer: The generated answer.
        reference: The reference (gold) answer.
        provider: The LLM provider acting as judge.

    Returns:
        The correctness judgment.
    """
    user = (
        f"PREGUNTA:\n{question}\n\nRESPUESTA DE REFERENCIA:\n{reference}\n\n"
        f"RESPUESTA:\n{answer}\n\nEvalúa la corrección."
    )
    reply = provider.complete(
        [
            ChatMessage(role="system", content=_CORRECTNESS_SYSTEM),
            ChatMessage(role="user", content=user),
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return _parse_judgment("correctness", reply)
