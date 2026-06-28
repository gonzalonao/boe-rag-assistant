"""Byte-identical text equivalence for fair retrieval scoring.

Some BOE passages are byte-identical across the corpus: a standard clause (e.g.
the tax-exclusion paragraph of a periodic LPG price resolution) is reproduced
verbatim in many separate documents. An embedding model scores every copy
identically, so which one wins the rank is an arbitrary tie-break. When a gold
label names exactly one copy, that tie can read as a miss even though an
interchangeable, byte-identical passage was retrieved — an artifact of the label,
not a retrieval failure (it is what pinned the embedding fine-tune's recall at
delta zero).

This module collapses byte-identical passages into equivalence classes so the
existing ranking metrics score *distinct content*: each class is represented by a
canonical chunk id (the lexicographically smallest member). Mapping both the
retrieved ranking and the relevant set onto canonical ids — and de-duplicating
the ranking — lets a hit on any class member count once, with the recall
denominator counting information needs (classes), not raw copies. It is a pure,
corpus-derived transform: the corpus itself is never modified, since the repeated
clauses are legitimate content of their own documents.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextEquivalence:
    """A map from chunk id to the canonical id of its byte-identical class.

    Only chunk ids that share their text with at least one other chunk are stored;
    every other id (singletons, and ids absent from the corpus) is its own
    representative. Construct via :func:`build_text_equivalence`.

    Attributes:
        canonical: Mapping of each non-representative chunk id to the canonical
            id of its equivalence class.
    """

    canonical: Mapping[str, str]

    def representative(self, chunk_id: str) -> str:
        """Return the canonical id of ``chunk_id``'s class (itself if unique)."""
        return self.canonical.get(chunk_id, chunk_id)

    def canonical_set(self, chunk_ids: Iterable[str]) -> frozenset[str]:
        """Map ids to their canonical representatives, collapsing duplicates."""
        return frozenset(self.representative(cid) for cid in chunk_ids)

    def canonical_sequence(self, chunk_ids: Iterable[str]) -> list[str]:
        """Map a ranking to canonical ids, keeping the first hit of each class.

        Preserves rank order; a run of byte-identical passages collapses to the
        single rank at which the class first appears, so duplicate content does
        not consume multiple rank positions.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for cid in chunk_ids:
            rep = self.representative(cid)
            if rep not in seen:
                seen.add(rep)
                ordered.append(rep)
        return ordered

    @property
    def num_redundant(self) -> int:
        """Number of chunk ids that are non-canonical duplicates of another."""
        return len(self.canonical)


def build_text_equivalence(
    chunk_ids: Sequence[str], texts: Sequence[str]
) -> TextEquivalence:
    """Build text-equivalence classes from a corpus' chunk ids and texts.

    Passages with byte-identical text form one class, represented by the
    lexicographically smallest chunk id in the class.

    Args:
        chunk_ids: Corpus chunk ids, aligned with ``texts``.
        texts: Corpus chunk texts, aligned with ``chunk_ids``.

    Returns:
        A :class:`TextEquivalence` mapping each duplicate id to its canonical id.

    Raises:
        ValueError: If ``chunk_ids`` and ``texts`` differ in length.
    """
    if len(chunk_ids) != len(texts):
        raise ValueError(
            f"chunk_ids and texts must have the same length, "
            f"got {len(chunk_ids)} and {len(texts)}"
        )
    by_text: dict[str, list[str]] = defaultdict(list)
    for chunk_id, text in zip(chunk_ids, texts, strict=True):
        by_text[text].append(chunk_id)

    canonical: dict[str, str] = {}
    for ids in by_text.values():
        if len(ids) < 2:
            continue
        representative = min(ids)
        for chunk_id in ids:
            if chunk_id != representative:
                canonical[chunk_id] = representative
    return TextEquivalence(canonical)
