"""Baseline grounded answer generation.

The naive RAG generator for the Phase 2 baseline: given a question and the
retrieved passages, prompt the LLM to answer *only* from those passages, cite
them as ``[n]``, and refuse when the answer is not present. Deliberately simple
— it is the "before" generator that Phase 5 will improve on.
"""

from __future__ import annotations

from collections.abc import Sequence

from boe_rag.llm.base import ChatMessage, LLMProvider

#: Exact refusal string the model is told to emit when the answer is absent.
REFUSAL = "No tengo información suficiente para responder."

_SYSTEM = (
    "Eres un asistente jurídico que responde preguntas sobre legislación "
    "española. Responde ÚNICAMENTE con la información de los fragmentos "
    "proporcionados y cita las fuentes que uses con su número entre corchetes, "
    f"por ejemplo [1]. Si la respuesta no aparece en los fragmentos, responde "
    f"exactamente: '{REFUSAL}'"
)


def build_context_block(contexts: Sequence[tuple[str, str]]) -> str:
    """Render retrieved passages as a numbered context block.

    Args:
        contexts: ``(citation, text)`` pairs in retrieval order.

    Returns:
        A newline-separated block of ``[n] (citation) text`` lines.
    """
    return "\n".join(
        f"[{i}] ({citation}) {text}"
        for i, (citation, text) in enumerate(contexts, start=1)
    )


def generate_answer(
    question: str,
    contexts: Sequence[tuple[str, str]],
    provider: LLMProvider,
) -> str:
    """Generate a grounded, cited answer for a question.

    Args:
        question: The user question.
        contexts: ``(citation, text)`` pairs retrieved for the question.
        provider: The LLM provider to use.

    Returns:
        The generated answer (or the refusal string when unsupported).
    """
    user = (
        f"Fragmentos:\n{build_context_block(contexts)}\n\n"
        f"Pregunta: {question}\n\nRespuesta:"
    )
    messages = [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    return provider.complete(messages, temperature=0.0, max_tokens=512)
