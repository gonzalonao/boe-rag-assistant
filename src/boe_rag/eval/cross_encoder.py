"""Cross-encoder reranker for the second retrieval stage.

Isolated in its own module so the heavy, untyped ``sentence-transformers``
dependency (the ``ml`` extra) is only imported when a real rerank run is
requested. The rest of the eval package depends only on the
:class:`~boe_rag.eval.rerank.Reranker` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Default reranker: multilingual MiniLM cross-encoder trained on mMARCO
#: (includes Spanish), small enough to rerank a ~30-doc pool on CPU.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    """Reranks candidates with a sentence-transformers cross-encoder.

    Args:
        model_name: Hugging Face cross-encoder model id to load.
        device: Torch device (e.g. ``"cpu"`` or ``"cuda"``); auto-selected when
            omitted.
    """

    def __init__(
        self, model_name: str = DEFAULT_RERANK_MODEL, device: str | None = None
    ) -> None:
        """Load the cross-encoder model."""
        # Imported lazily so the package works without the `ml` extra installed.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Score each candidate passage against the query and sort by score.

        Args:
            query: The search query.
            candidates: ``(chunk_id, text)`` pairs to rerank.

        Returns:
            ``(chunk_id, score)`` pairs sorted by descending relevance.
        """
        if not candidates:
            return []
        pairs = [[query, text] for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = [
            (cid, float(score))
            for (cid, _), score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
