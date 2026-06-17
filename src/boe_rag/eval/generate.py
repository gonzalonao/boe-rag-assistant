"""LLM-assisted generation of golden eval questions from corpus chunks.

The hand-curated seed set is small; this module grows it by prompting an LLM to
write a self-contained question (plus a reference answer) whose answer lives in a
given chunk. The chunk it was generated from is, by construction, the relevant
chunk — so each generated item is a ``(question, answer, relevant_chunk_id)``
triple ready for the eval harness.

Quality is enforced in two cheap stages here, with an optional LLM faithfulness
filter applied by the calling script:

* the prompt forbids deictic references ("este artículo", "el fragmento"), and
* :func:`is_self_contained` drops any question that slips through or is too short.

Generation must stay grounded; the question has to make sense on its own, the way
a real user would ask it without seeing the source text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider

#: First JSON object embedded anywhere in a model reply.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Minimum number of words for a question to count as substantive.
MIN_QUESTION_WORDS = 5

#: Phrases that reveal a question only makes sense next to its source text.
_DEICTIC_MARKERS: tuple[str, ...] = (
    "fragmento",
    "este artículo",
    "dicho artículo",
    "presente artículo",
    "citado artículo",
    "el texto",
    "este texto",
    "este documento",
    "documento mostrado",
    "anteriormente",
    "lo anterior",
    "mostrado",
    "según el extracto",
)

_GENERATION_SYSTEM = (
    "Eres un experto jurídico que crea preguntas de evaluación sobre legislación "
    "española. A partir del FRAGMENTO de un documento del BOE, redacta UNA "
    "pregunta clara y autosuficiente que un ciudadano o jurista podría plantear y "
    "cuya respuesta esté contenida en el fragmento, junto con una respuesta breve "
    "y precisa. La pregunta debe entenderse por sí sola: no puede referirse a "
    "'este fragmento', 'el texto', 'este artículo' ni a lo mostrado, y debe "
    "mencionar el tema o la norma concreta a la que se refiere. Devuelve SOLO un "
    'objeto JSON con las claves "pregunta" y "respuesta".'
)


@dataclass(frozen=True, slots=True)
class GeneratedQA:
    """A generated question and its reference answer.

    Attributes:
        question: The generated, self-contained question.
        answer: The reference answer drawn from the source chunk.
    """

    question: str
    answer: str


def build_generation_messages(chunk_text: str, citation: str) -> list[ChatMessage]:
    """Build the chat messages that prompt for a question/answer pair.

    Args:
        chunk_text: The source chunk's text.
        citation: The chunk's human-readable citation (e.g. law and article).

    Returns:
        The system and user messages for the generation call.
    """
    user = (
        f"FRAGMENTO ({citation}):\n{chunk_text}\n\nGenera la pregunta y la respuesta."
    )
    return [
        ChatMessage(role="system", content=_GENERATION_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def parse_generated_qa(raw: str) -> GeneratedQA:
    """Parse a model reply into a :class:`GeneratedQA`.

    Args:
        raw: The model's raw reply (may include prose or code fences).

    Returns:
        The parsed question/answer pair.

    Raises:
        LLMError: If no question/answer JSON can be parsed from the reply.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise LLMError(f"generation reply has no JSON object: {raw!r}")
    try:
        payload = json.loads(match.group(0))
        question = str(payload["pregunta"]).strip()
        answer = str(payload["respuesta"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        raise LLMError(f"unparseable generation reply: {raw!r}") from err
    if not question or not answer:
        raise LLMError(f"generation reply missing question or answer: {raw!r}")
    return GeneratedQA(question=question, answer=answer)


def is_self_contained(question: str) -> bool:
    """Whether a question stands on its own (no deictic refs, long enough).

    Args:
        question: The question to check.

    Returns:
        ``True`` if the question is substantive and free of references to the
        source text, ``False`` otherwise.
    """
    lowered = question.lower()
    if len(question.split()) < MIN_QUESTION_WORDS:
        return False
    return not any(marker in lowered for marker in _DEICTIC_MARKERS)


def generate_qa(chunk_text: str, citation: str, provider: LLMProvider) -> GeneratedQA:
    """Generate a question/answer pair grounded in a chunk.

    Args:
        chunk_text: The source chunk's text.
        citation: The chunk's human-readable citation.
        provider: The LLM provider to use.

    Returns:
        The generated question/answer pair.

    Raises:
        LLMError: If the call fails or the reply cannot be parsed.
    """
    reply = provider.complete(
        build_generation_messages(chunk_text, citation),
        temperature=0.3,
        max_tokens=400,
    )
    return parse_generated_qa(reply)
