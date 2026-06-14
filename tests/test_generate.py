"""Tests for LLM-assisted eval-question generation, using a fake provider."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from boe_rag.eval.generate import (
    GeneratedQA,
    generate_qa,
    is_self_contained,
    parse_generated_qa,
)
from boe_rag.llm.base import ChatMessage, LLMError


class _ScriptedProvider:
    """Provider that returns a fixed reply, recording the prompt it saw."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_messages: Sequence[ChatMessage] | None = None

    @property
    def name(self) -> str:
        return "scripted"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.last_messages = messages
        return self._reply


def test_parse_generated_qa_extracts_pair() -> None:
    """A JSON reply (even fenced) yields the question/answer pair."""
    raw = (
        "```json\n"
        '{"pregunta": "¿Qué regula la Ley 39/2015?", "respuesta": "El procedimiento."}'
        "\n```"
    )
    qa = parse_generated_qa(raw)
    assert qa == GeneratedQA(
        question="¿Qué regula la Ley 39/2015?", answer="El procedimiento."
    )


def test_parse_generated_qa_rejects_missing_keys() -> None:
    """A reply without both keys is an error."""
    with pytest.raises(LLMError):
        parse_generated_qa('{"pregunta": "solo pregunta"}')


def test_parse_generated_qa_rejects_non_json() -> None:
    """A reply with no JSON object is an error."""
    with pytest.raises(LLMError):
        parse_generated_qa("no hay json aquí")


def test_is_self_contained_accepts_specific_question() -> None:
    """A specific, standalone question is accepted."""
    assert is_self_contained("¿Qué plazo establece la Ley 39/2015 para resolver?")


def test_is_self_contained_rejects_deictic_question() -> None:
    """A question referring to the source text is rejected."""
    assert not is_self_contained("¿Qué establece este artículo sobre los plazos?")
    assert not is_self_contained("¿Qué dice el fragmento anterior?")


def test_is_self_contained_rejects_too_short() -> None:
    """A question with too few words is rejected."""
    assert not is_self_contained("¿Qué regula?")


def test_generate_qa_uses_provider_and_parses() -> None:
    """generate_qa sends the chunk to the provider and parses its reply."""
    provider = _ScriptedProvider(
        '{"pregunta": "¿Cuál es el tipo del IVA reducido?", "respuesta": "El 10%."}'
    )
    qa = generate_qa("El IVA reducido es del 10%.", "Ley 37/1992", provider)
    assert qa.question.startswith("¿Cuál es el tipo")
    assert provider.last_messages is not None
    # The chunk text and citation reach the model.
    assert "Ley 37/1992" in provider.last_messages[-1].content
