"""Post-hoc citation validation for grounded answers.

The cite-or-refuse system prompt cannot *guarantee* citation integrity: the
adversarial security eval (``boe_rag.eval.security``) found that, when goaded, the
generator will cite passages it was never given — e.g. ``[99]`` against a five-source
context, or echo an injected literal with no citation at all. Prompt-level defenses
caught the fabrication 0% of the time.

This module is the deterministic guardrail that runs *after* generation, enforcing a
single **cite-or-refuse** invariant on the output: a served answer must carry at least
one *valid* citation, otherwise it is replaced by the refusal string. A citation index
is valid only when it falls in ``1..num_sources`` (it points at a passage actually
supplied to the generator).

Two failure modes collapse to a refusal under this rule:

- **Fabricated grounding** — the answer cites only indices it was never given
  (``[99]``); the markers are meaningless, so the answer has nothing to stand on.
- **Uncited output** — the answer carries no ``[n]`` at all. The generator is
  instructed to cite every claim, so an uncited non-refusal is either a malfunction or
  an injected echo (e.g. an instruction-override payload), which by construction never
  carries a real citation.

When an answer *does* carry a valid citation, any *additional* fabricated markers are
stripped and the answer is served — a partially-grounded answer keeps its grounding.
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
        uncited: Whether the answer carried no ``[n]`` citation at all. ``True`` only
            on the refusal path, where it distinguishes an uncited echo from an
            answer whose every citation was fabricated (for tracing/triage).
    """

    answer: str
    refused: bool
    invalid_citations: tuple[int, ...]
    uncited: bool = False


def cited_indices(answer: str) -> list[int]:
    """Return the ``[n]`` citation indices in an answer, in order (with repeats)."""
    return [int(match) for match in _CITATION_RE.findall(answer)]


def validate_citations(
    answer: str, num_sources: int, *, refusal: str
) -> CitationValidation:
    """Enforce cite-or-refuse: a served answer must carry a valid citation.

    An index is valid only inside ``1..num_sources``; any other index (zero,
    negative, or past the last retrieved passage) is a fabrication. The answer is
    served **iff** it cites at least one valid index, in which case any fabricated
    markers are stripped from the text. Otherwise — whether it cited only fabricated
    sources or cited nothing at all — it has no grounding in the retrieved passages
    and is replaced by ``refusal``.

    Args:
        answer: The generated answer text.
        num_sources: How many passages were supplied to the generator.
        refusal: The exact refusal string to emit when grounding collapses.

    Returns:
        The validated answer, whether it was refused, the stripped indices, and
        whether the refusal was triggered by an uncited answer.
    """
    indices = cited_indices(answer)
    invalid = tuple(i for i in indices if i < 1 or i > num_sources)
    has_valid = any(1 <= i <= num_sources for i in indices)

    if not has_valid:
        # No grounding the answer can stand on: every citation (if any) was
        # fabricated, or there was none at all. Either way, refuse.
        return CitationValidation(
            answer=refusal,
            refused=True,
            invalid_citations=invalid,
            uncited=not indices,
        )

    cleaned = _strip_invalid(answer, num_sources) if invalid else answer
    return CitationValidation(answer=cleaned, refused=False, invalid_citations=invalid)


def _strip_invalid(answer: str, num_sources: int) -> str:
    """Remove ``[n]`` markers whose index falls outside ``1..num_sources``."""

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= num_sources else ""

    return _CITATION_RE.sub(replace, answer).strip()
