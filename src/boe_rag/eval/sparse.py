"""Pure-Python BM25 sparse retriever.

A lexical counterpart to the dense retriever: it ranks chunks by Okapi BM25 over
an inverted index, so exact legal terms ("artículo 14", "Ley 39/2015", named
entities) are matched directly rather than approximated by embeddings. It has no
heavy dependencies, so — like the metrics — it runs in CI and is the sparse leg
of the hybrid retriever.

The tokenizer is intentionally simple and deterministic: lowercase, split on
non-alphanumeric boundaries (keeping Spanish accented letters and ``ñ``), and
drop a small set of high-frequency Spanish stopwords. Accents are preserved
because both the corpus and the eval questions are properly accented.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

#: Default Okapi BM25 term-frequency saturation parameter.
DEFAULT_K1 = 1.5
#: Default Okapi BM25 length-normalisation parameter.
DEFAULT_B = 0.75

#: Matches runs of digits and (accented) letters; everything else is a boundary.
_TOKEN_RE = re.compile(r"[0-9a-záéíóúüñ]+")

#: High-frequency Spanish function words that carry little retrieval signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "al",
        "ante",
        "como",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "donde",
        "durante",
        "e",
        "el",
        "en",
        "entre",
        "era",
        "es",
        "esa",
        "ese",
        "eso",
        "esta",
        "este",
        "esto",
        "ha",
        "han",
        "hasta",
        "la",
        "las",
        "lo",
        "los",
        "mas",
        "más",
        "me",
        "mi",
        "mucho",
        "muy",
        "no",
        "nos",
        "o",
        "para",
        "pero",
        "por",
        "porque",
        "que",
        "qué",
        "se",
        "según",
        "sea",
        "ser",
        "si",
        "sí",
        "sin",
        "so",
        "sobre",
        "son",
        "su",
        "sus",
        "tan",
        "te",
        "tu",
        "un",
        "una",
        "uno",
        "unos",
        "unas",
        "y",
        "ya",
    }
)


def tokenize_es(text: str) -> list[str]:
    """Tokenise Spanish text for lexical matching.

    Args:
        text: Raw text to tokenise.

    Returns:
        Lowercased single-token terms with stopwords and 1-character tokens
        removed, in order of appearance.
    """
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


class BM25Index:
    """In-memory Okapi BM25 retriever over a tokenised corpus.

    Args:
        k1: Term-frequency saturation parameter.
        b: Document-length normalisation parameter (0 = none, 1 = full).
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        """Create an empty index with the given BM25 hyperparameters."""
        self._k1 = k1
        self._b = b
        self._chunk_ids: list[str] = []
        self._doc_len: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._avgdl = 0.0
        #: term -> list of ``(doc_index, term_frequency)`` postings.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        #: term -> inverse document frequency.
        self._idf: dict[str, float] = {}

    def index(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Tokenise and index the corpus.

        Args:
            chunk_ids: Stable ids, aligned with ``texts``.
            texts: Chunk texts to index.

        Raises:
            ValueError: If the inputs are empty or of unequal length.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        if not chunk_ids:
            raise ValueError("cannot index an empty corpus")

        self._chunk_ids = list(chunk_ids)
        n_docs = len(chunk_ids)
        doc_len = np.zeros(n_docs, dtype=np.float64)
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: dict[str, int] = {}

        for doc_idx, text in enumerate(texts):
            tokens = tokenize_es(text)
            doc_len[doc_idx] = len(tokens)
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            for term, freq in term_freqs.items():
                postings.setdefault(term, []).append((doc_idx, freq))
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self._doc_len = doc_len
        self._avgdl = float(doc_len.mean()) if n_docs else 0.0
        self._postings = postings
        self._idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k chunk ids and BM25 scores for a query.

        Args:
            query: The search query.
            k: Maximum number of results to return.

        Returns:
            Up to ``k`` ``(chunk_id, score)`` pairs with a positive score,
            highest score first. Chunks matching no query term are excluded.

        Raises:
            RuntimeError: If called before :meth:`index`.
        """
        if not self._chunk_ids:
            raise RuntimeError("call index() before search()")

        scores = np.zeros(len(self._chunk_ids), dtype=np.float64)
        length_norm = self._k1 * (1.0 - self._b + self._b * self._doc_len / self._avgdl)
        for term in set(tokenize_es(query)):
            postings = self._postings.get(term)
            if postings is None:
                continue
            idf = self._idf[term]
            for doc_idx, freq in postings:
                scores[doc_idx] += idf * (
                    freq * (self._k1 + 1.0) / (freq + length_norm[doc_idx])
                )

        top_k = min(k, len(self._chunk_ids))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            (self._chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0.0
        ]
