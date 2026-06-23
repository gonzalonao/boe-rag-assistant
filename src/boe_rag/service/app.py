"""Serving entrypoint: assemble the real engine and expose the ASGI ``app``.

Run with ``uvicorn boe_rag.service.app:app`` (the ``api`` + ``ui`` extras). This
module loads the corpus and the embedding/rerank models at import time, so it is
the production wiring — never imported by the unit tests, which inject a fake
engine. The Gradio demo UI is mounted at ``/`` and the JSON API lives alongside
it (``/ask``, ``/search``, ``/health``, ``/docs``).

Configuration via environment (or a local ``.env`` — see ``.env.example`` and
:func:`boe_rag.settings.load_environment`; real environment variables win):
    ``BOE_CORPUS_PATH``  path to the corpus Parquet (default the bundled 2024 set).
    ``BOE_EMBEDDINGS_PATH``  optional precomputed-embeddings ``.npz`` to skip the
        startup encode (see ``scripts/precompute_embeddings.py``).
    ``BOE_REPORTS_DIR``  directory of eval report JSON for the Quality tab.
    ``OPENROUTER_API_KEY`` / ``GROQ_API_KEY`` / ``GEMINI_API_KEY``  at least one is
        required for ``/ask`` (tried in that order; the first with a key leads).
    ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` (+ optional ``LANGFUSE_HOST``)
        opt-in: when both are set, pipeline stages are traced to Langfuse (needs
        the ``obs`` extra); otherwise tracing is a no-op.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr
import pyarrow.parquet as pq

from boe_rag.eval.cross_encoder import CrossEncoderReranker
from boe_rag.eval.embedding import E5Embedder
from boe_rag.eval.hybrid import HybridRetriever
from boe_rag.eval.qdrant_store import connect_searcher
from boe_rag.eval.retriever import DenseRetriever, Searcher, load_embeddings
from boe_rag.eval.sparse import BM25Index
from boe_rag.llm.base import LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.service.api import create_app
from boe_rag.service.engine import ChunkInfo, RagEngine
from boe_rag.service.tracing import build_tracer
from boe_rag.service.ui import build_ui, render_quality_markdown
from boe_rag.settings import load_environment

logger = logging.getLogger(__name__)

#: Default corpus location when ``BOE_CORPUS_PATH`` is unset.
DEFAULT_CORPUS_PATH = Path("data/corpus/boe-2024.parquet")
#: Default directory of eval report JSON when ``BOE_REPORTS_DIR`` is unset.
DEFAULT_REPORTS_DIR = Path("reports")

#: Settings loaded once at import: reads the environment plus an optional ``.env``
#: and exports it, so provider/tracing lookups below see ``.env`` values too.
_SETTINGS = load_environment()


def _index_dense(ids: list[str], texts: list[str]) -> DenseRetriever:
    """Build the dense retriever, loading precomputed embeddings if available.

    When ``BOE_EMBEDDINGS_PATH`` points at a ``.npz`` produced by
    ``scripts/precompute_embeddings.py``, the corpus is not re-encoded at boot —
    the matrix is loaded directly (the embedder is kept only for query encoding).
    The precomputed ids must match the corpus order; otherwise we fall back to
    encoding so a stale file can never serve wrong results.
    """
    dense = DenseRetriever(E5Embedder())
    embeddings_path = _SETTINGS.embeddings_path
    if embeddings_path and embeddings_path.is_file():
        cached_ids, matrix = load_embeddings(embeddings_path)
        if cached_ids == ids:
            logger.info("Loading %d precomputed embeddings ...", len(cached_ids))
            dense.index_precomputed(cached_ids, matrix)
            return dense
        logger.warning(
            "Precomputed embeddings at %s do not match the corpus; re-encoding.",
            embeddings_path,
        )
    dense.index(ids, texts)
    return dense


def _build_dense_leg(ids: list[str], texts: list[str]) -> Searcher:
    """Build the dense retrieval leg, preferring Qdrant when configured.

    When ``QDRANT_COLLECTION`` and either ``QDRANT_URL`` (a server) or
    ``QDRANT_PATH`` (a local embedded instance) are set, the dense leg is served
    from a Qdrant collection — assumed already populated by
    ``scripts/build_qdrant_index.py`` from the same embeddings — instead of the
    in-memory NumPy index. The collection must hold vectors from the same model
    :class:`E5Embedder` queries with. Otherwise the NumPy path is used unchanged.
    """
    url = _SETTINGS.qdrant_url
    path = _SETTINGS.qdrant_path
    collection = _SETTINGS.qdrant_collection
    if (url or path) and collection:
        logger.info(
            "Serving the dense leg from Qdrant '%s' at %s",
            collection,
            url or path,
        )
        return connect_searcher(
            collection,
            E5Embedder(),
            url=url,
            path=path,
            api_key=_SETTINGS.qdrant_api_key,
        )
    return _index_dense(ids, texts)


def _load_corpus(
    path: Path,
) -> tuple[list[str], list[str], dict[str, ChunkInfo]]:
    """Load chunk ids/texts and a chunk-id → :class:`ChunkInfo` lookup."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path, columns=["chunk_id", "text", "citation", "url_html"]
    )
    data = table.to_pydict()
    ids = list(map(str, data["chunk_id"]))
    texts = list(map(str, data["text"]))
    citations = list(map(str, data["citation"]))
    urls = list(map(str, data["url_html"]))
    lookup = {
        cid: ChunkInfo(citation=cit, text=txt, url=url)
        for cid, txt, cit, url in zip(ids, texts, citations, urls, strict=True)
    }
    return ids, texts, lookup


def build_engine(corpus_path: Path | None = None) -> RagEngine:
    """Assemble the production RAG engine from the corpus and models.

    Args:
        corpus_path: Corpus Parquet path; defaults to ``BOE_CORPUS_PATH`` or the
            bundled location.

    Returns:
        A ready-to-serve :class:`RagEngine` (hybrid retrieval + cross-encoder
        rerank + grounded generation).

    Raises:
        LLMError: If no LLM provider is configured.
    """
    path = corpus_path or _SETTINGS.corpus_path or DEFAULT_CORPUS_PATH
    providers = build_available_providers()
    if not providers:
        raise LLMError(
            "No LLM provider configured. Set OPENROUTER_API_KEY (recommended) "
            "and/or GROQ_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY."
        )

    logger.info("Loading corpus from %s ...", path)
    ids, texts, lookup = _load_corpus(path)

    logger.info("Indexing %d chunks ...", len(ids))
    dense = _build_dense_leg(ids, texts)
    bm25 = BM25Index()
    bm25.index(ids, texts)
    hybrid = HybridRetriever(dense, bm25)

    return RagEngine(
        retriever=hybrid,
        lookup=lookup,
        provider=FallbackProvider(providers),
        reranker=CrossEncoderReranker(),
        tracer=build_tracer(),
    )


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object from ``path``, or an empty dict if it is missing."""
    if not path.is_file():
        logger.warning("Report not found, Quality tab will note it: %s", path)
        return {}
    with path.open(encoding="utf-8") as handle:
        data: dict[str, object] = json.load(handle)
    return data


def _quality_markdown(reports_dir: Path | None = None) -> str:
    """Render the Quality-tab Markdown from the eval report JSON files."""
    directory = reports_dir or _SETTINGS.reports_dir or DEFAULT_REPORTS_DIR
    retrieval = _load_json(directory / "retrieval_rerank.json")
    e2e = _load_json(directory / "e2e_baseline.json")
    return render_quality_markdown(retrieval, e2e)  # type: ignore[arg-type]


def build_app(corpus_path: Path | None = None) -> object:
    """Assemble the FastAPI app with the Gradio demo UI mounted at ``/``.

    Args:
        corpus_path: Corpus Parquet path; defaults to the env/bundled location.

    Returns:
        The ASGI application (FastAPI with Gradio mounted at the root).
    """
    engine = build_engine(corpus_path)
    api = create_app(engine)
    demo = build_ui(engine, _quality_markdown())
    return gr.mount_gradio_app(api, demo, path="/")


app = build_app()
