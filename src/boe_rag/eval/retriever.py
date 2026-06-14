"""Baseline dense retriever.

The Phase 2 baseline: embed every chunk with an off-the-shelf multilingual
model and rank by cosine similarity — no hybrid search, no reranking. It is the
"before" picture every later retrieval improvement is measured against.

The embedding model is injected behind the :class:`Embedder` protocol so the
ranking logic can be unit-tested with a trivial fake embedder, keeping the heavy
``sentence-transformers``/``torch`` dependency out of CI.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

#: A matrix of L2-normalised row vectors.
FloatMatrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Turns text into L2-normalised embedding vectors.

    Implementations must return one row per input text. Queries and passages
    are embedded via separate methods because some models (e.g. E5) require
    different prefixes for each.
    """

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed documents/passages to be indexed."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed search queries."""
        ...


class DenseRetriever:
    """In-memory dense retriever using cosine similarity over chunk embeddings.

    Args:
        embedder: The embedder used for both indexing and querying.
    """

    def __init__(self, embedder: Embedder) -> None:
        """Create an empty retriever bound to ``embedder``."""
        self._embedder = embedder
        self._chunk_ids: list[str] = []
        self._matrix: FloatMatrix | None = None

    def index(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Embed and store the corpus.

        Args:
            chunk_ids: Stable ids, aligned with ``texts``.
            texts: Chunk texts to embed and index.

        Raises:
            ValueError: If the inputs are empty or of unequal length.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        if not chunk_ids:
            raise ValueError("cannot index an empty corpus")
        self._chunk_ids = list(chunk_ids)
        self._matrix = self._embedder.embed_passages(list(texts))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k chunk ids and similarity scores for a query.

        Args:
            query: The search query.
            k: Maximum number of results to return.

        Returns:
            Up to ``k`` ``(chunk_id, score)`` pairs, highest score first.

        Raises:
            RuntimeError: If called before :meth:`index`.
        """
        if self._matrix is None:
            raise RuntimeError("call index() before search()")
        query_vec = self._embedder.embed_queries([query])[0]
        scores = self._matrix @ query_vec
        top_k = min(k, len(self._chunk_ids))
        # argpartition for the top-k, then sort just those by score descending.
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_idx]
