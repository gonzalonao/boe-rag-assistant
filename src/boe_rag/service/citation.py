"""Post-hoc citation validation for grounded answers.

The cite-or-refuse system prompt cannot *guarantee* citation integrity: the
adversarial security eval (``boe_rag.eval.security``) found that, when goaded, the
generator will cite passages it was never given — e.g. ``[99]`` against a five-source
context. Prompt-level defenses caught it 0% of the time.

This module is the deterministic guardrail that runs *after* generation. A citation
index is valid only when it falls in ``1..num_sources`` (it points at a passage
actually supplied to the generator). Fabricated markers are stripped from the text,
and if an answer's grounding rested *entirely* on fabricated citations it is replaced
by the refusal string — an answer that cites only sources it never saw has no real
grounding to stand on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Matches a citation marker and any single space immediately before it, so that
#: stripping a marker does not leave a dangling double space.
_CITATION_RE = re.compile(r" ?\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """Outcome of validating an answer's citations.

    Attributes:
        answer: The answer text with fabricated citation markers removed (or the
            refusal string when grounding collapsed).
        refused: Whether the answer was rejected for lacking any valid grounding.
        invalid_citations: The fabricated indices that were stripped, in order of
            appearance (with repeats).
    """

    answer: str
    refused: bool
    invalid_citations: tuple[int, ...]


def cited_indices(answer: str) -> list[int]:
    """Return the ``[n]`` citation indices in an answer, in order (with repeats)."""
    return [int(match) for match in _CITATION_RE.findall(answer)]


def validate_citations(
    answer: str, num_sources: int, *, refusal: str
) -> CitationValidation:
    """Strip fabricated ``[n]`` citations, refusing when none remain valid.

    An index is valid only inside ``1..num_sources``. Any other index (zero,
    negative, or past the last retrieved passage) is a fabrication: its marker is
    removed from the text. If the answer cited sources but *every* citation was
    fabricated, the answer has no genuine grounding and is replaced by ``refusal``.

    Args:
        answer: The generated answer text.
        num_sources: How many passages were supplied to the generator.
        refusal: The exact refusal string to emit when grounding collapses.

    Returns:
        The validated answer, whether it was refused, and the stripped indices.
    """
    indices = cited_indices(answer)
    invalid = tuple(i for i in indices if i < 1 or i > num_sources)
    if not invalid:
        return CitationValidation(answer=answer, refused=False, invalid_citations=())

    has_valid = any(1 <= i <= num_sources for i in indices)
    if not has_valid:
        return CitationValidation(
            answer=refusal, refused=True, invalid_citations=invalid
        )

    cleaned = _strip_invalid(answer, num_sources)
    return CitationValidation(answer=cleaned, refused=False, invalid_citations=invalid)


def _strip_invalid(answer: str, num_sources: int) -> str:
    """Remove ``[n]`` markers whose index falls outside ``1..num_sources``."""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= num_sources else ""

    return _CITATION_RE.sub(replace, answer).strip()
