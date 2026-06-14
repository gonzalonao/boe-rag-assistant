"""Sentence-Transformers embedder for the baseline retriever.

Isolated in its own module so the heavy, untyped ``sentence-transformers``
dependency is only imported when an actual embedding run is requested (it lives
behind the optional ``ml`` extra). The rest of the eval package depends only on
the :class:`~boe_rag.eval.retriever.Embedder` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from boe_rag.eval.retriever import FloatMatrix

#: Default baseline model: small, multilingual, strong on Spanish, CPU-friendly.
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class E5Embedder:
    """E5-family embedder that applies the required query/passage prefixes.

    E5 models are trained with ``"query: "`` and ``"passage: "`` prefixes and
    expect L2-normalised embeddings for cosine similarity; both are handled here.

    Args:
        model_name: Hugging Face model id to load.
        device: Torch device (e.g. ``"cpu"`` or ``"cuda"``); auto-selected when
            omitted.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str | None = None
    ) -> None:
        """Load the sentence-transformers model."""
        # Imported lazily so the package works without the `ml` extra installed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: Sequence[str], prefix: str) -> FloatMatrix:
        """Encode texts with the given E5 prefix into a normalised matrix."""
        prefixed = [f"{prefix}{text}" for text in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed passages with the ``passage:`` prefix."""
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed queries with the ``query:`` prefix."""
        return self._encode(texts, "query: ")
