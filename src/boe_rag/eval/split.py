"""Leakage-free train/test split of an eval set by source document.

The go/no-go gate (:mod:`boe_rag.eval.compare`) needs a held-out test set with
enough queries to detect a real retrieval gain — the 20-query gold set is too
small, its recall@k pinned by a couple of structurally hard queries. The silver
set (~1.7k LLM-generated questions) has the volume, so we carve a held-out test
split from it.

A naive per-row split would leak: two questions can point at the *same* BOE
document (its chunks share boilerplate and near-identical phrasing), so training
on one and testing on the other lets the model "see" the answer. This module
splits by **document** instead — examples whose positives come from the same
document, or from documents linked through a shared example, are kept together on
one side of the split (a union-find over co-occurring documents). The result is a
test set whose source documents never appear in training.

The split is deterministic given the seed, so it need not be committed: the same
seed reproduces the same partition.
"""

from __future__ import annotations

import random
from collections import defaultdict

from boe_rag.eval.dataset import EvalExample

#: Separator between the document id and the section index in a chunk id, e.g.
#: ``BOE-A-2024-714::0004`` -> document ``BOE-A-2024-714``, section ``0004``.
_CHUNK_ID_SEP = "::"


def document_id(chunk_id: str) -> str:
    """Return the source-document id of a chunk id (the part before ``::``).

    Args:
        chunk_id: A corpus chunk id, e.g. ``BOE-A-2024-714::0004``.

    Returns:
        The document id (``BOE-A-2024-714``); the whole id if no separator.
    """
    return chunk_id.split(_CHUNK_ID_SEP, 1)[0]


class _DisjointSet:
    """Minimal union-find over string keys (path compression, union by size)."""

    def __init__(self) -> None:
        """Start with no elements."""
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, item: str) -> None:
        """Register ``item`` as its own singleton set if unseen."""
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        """Return the representative root of ``item``'s set."""
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point every node on the way up straight at the root.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing ``a`` and ``b`` (smaller into larger)."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]


def split_by_document(
    examples: list[EvalExample],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[EvalExample], list[EvalExample]]:
    """Partition examples into train/test with disjoint source documents.

    Examples are grouped so that any two sharing a source document — directly, or
    transitively through other examples — land on the same side. Whole groups are
    then assigned to the test split (in a seeded random order) until it reaches
    roughly ``test_fraction`` of the examples; the rest go to train. No document
    appears in both splits.

    Args:
        examples: The examples to partition (e.g. the loaded silver set).
        test_fraction: Target share of examples for the test split, in ``(0, 1)``.
        seed: Seed for the deterministic group ordering.

    Returns:
        A ``(train, test)`` pair of example lists, each in the input's order.

    Raises:
        ValueError: If ``test_fraction`` is not strictly between 0 and 1, or if
            ``examples`` is empty.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not examples:
        raise ValueError("cannot split an empty example list")

    # Link all documents that co-occur within a single example.
    dsu = _DisjointSet()
    for example in examples:
        docs = [document_id(cid) for cid in example.relevant_chunk_ids]
        for doc in docs:
            dsu.add(doc)
        for doc in docs[1:]:
            dsu.union(docs[0], doc)

    # Bucket examples by the root of their (first) document's group.
    groups: dict[str, list[EvalExample]] = defaultdict(list)
    for example in examples:
        root = dsu.find(document_id(example.relevant_chunk_ids[0]))
        groups[root].append(example)

    # Deterministically shuffle whole groups, then greedily fill the test split.
    roots = sorted(groups)
    random.Random(seed).shuffle(roots)
    target = round(len(examples) * test_fraction)

    test_ids: set[str] = set()
    test_count = 0
    for root in roots:
        if test_count >= target:
            break
        for example in groups[root]:
            test_ids.add(example.example_id)
        test_count += len(groups[root])

    train = [ex for ex in examples if ex.example_id not in test_ids]
    test = [ex for ex in examples if ex.example_id in test_ids]
    return train, test
