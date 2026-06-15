"""Serving entrypoint: assemble the real engine and expose the ASGI ``app``.

Run with ``uvicorn boe_rag.service.app:app`` (the ``api`` + ``ui`` extras). This
module loads the corpus and the embedding/rerank models at import time, so it is
the production wiring — never imported by the unit tests, which inject a fake
engine. The Gradio demo UI is mounted at ``/`` and the JSON API lives alongside
it (``/ask``, ``/search``, ``/health``, ``/docs``).

Configuration via environment:
    ``BOE_CORPUS_PATH``  path to the corpus Parquet (default the bundled 2024 set).
    ``BOE_REPORTS_DIR``  directory of eval report JSON for the Quality tab.
    ``GROQ_API_KEY`` / ``GEMINI_API_KEY``  at least one is required for ``/ask``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr
import pyarrow.parquet as pq

from boe_rag.eval.cross_encoder import CrossEncoderReranker
from boe_rag.eval.embedding import E5Embedder
from boe_rag.eval.hybrid import HybridRetriever
from boe_rag.eval.retriever import DenseRetriever
from boe_rag.eval.sparse import BM25Index
from boe_rag.llm.base import LLMError
from boe_rag.llm.factory import FallbackProvider, build_available_providers
from boe_rag.service.api import create_app
from boe_rag.service.engine import ChunkInfo, RagEngine
from boe_rag.service.ui import build_ui, render_quality_markdown

logger = logging.getLogger(__name__)

#: Default corpus location when ``BOE_CORPUS_PATH`` is unset.
DEFAULT_CORPUS_PATH = Path("data/corpus/boe-2024.parquet")
#: Default directory of eval report JSON when ``BOE_REPORTS_DIR`` is unset.
DEFAULT_REPORTS_DIR = Path("reports")


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
    path = corpus_path or Path(
        os.environ.get("BOE_CORPUS_PATH", str(DEFAULT_CORPUS_PATH))
    )
    providers = build_available_providers()
    if not providers:
        raise LLMError(
            "No LLM provider configured. Set GROQ_API_KEY (recommended) "
            "and/or GEMINI_API_KEY/GOOGLE_API_KEY."
        )

    logger.info("Loading corpus from %s ...", path)
    ids, texts, lookup = _load_corpus(path)

    logger.info("Indexing %d chunks ...", len(ids))
    dense = DenseRetriever(E5Embedder())
    dense.index(ids, texts)
    bm25 = BM25Index()
    bm25.index(ids, texts)
    hybrid = HybridRetriever(dense, bm25)

    return RagEngine(
        retriever=hybrid,
        lookup=lookup,
        provider=FallbackProvider(providers),
        reranker=CrossEncoderReranker(),
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
    directory = reports_dir or Path(
        os.environ.get("BOE_REPORTS_DIR", str(DEFAULT_REPORTS_DIR))
    )
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
