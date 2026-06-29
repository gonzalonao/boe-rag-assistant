"""FastAPI application exposing the RAG engine.

Endpoints: ``/health`` (readiness), ``/search`` (raw retrieval), and ``/ask``
(grounded, cited answer). The app is built from an injected :class:`Engine`, so
it is unit-tested with a fake engine — no models or API keys in CI. Identical
answers are cached, and a lightweight per-client rate limit protects the free
LLM tier. The UI is a separate single-page app (``frontend/``) that consumes
this API cross-origin; when a ``frontend_url`` is configured the API root
(``/``) redirects there, so the deployment URL still lands on the live UI.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse, RedirectResponse, Response

from boe_rag.llm.base import LLMError
from boe_rag.service.engine import Engine
from boe_rag.service.models import (
    AnswerResponse,
    AskRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

#: Max number of distinct answers cached in memory.
DEFAULT_CACHE_SIZE = 256
#: Max requests allowed per client within the rate-limit window.
DEFAULT_RATE_LIMIT = 60
#: Rate-limit window in seconds.
DEFAULT_RATE_WINDOW = 60.0
#: Only the expensive LLM/retrieval endpoints are rate-limited; readiness probes
#: and the root redirect are exempt so the UI and health checks are never blocked.
_RATE_LIMITED_PATHS = ("/ask", "/search")


class _AnswerCache:
    """A tiny bounded LRU cache for ``(question, k)`` answers."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[tuple[str, int], AnswerResponse] = OrderedDict()

    def get(self, key: tuple[str, int]) -> AnswerResponse | None:
        """Return a cached answer and mark it most-recently-used, or ``None``."""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: tuple[str, int], value: AnswerResponse) -> None:
        """Insert an answer, evicting the least-recently-used if over capacity."""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


class _RateLimiter:
    """Fixed-window per-client request limiter."""

    def __init__(self, limit: int, window: float) -> None:
        self._limit = limit
        self._window = window
        self._hits: dict[str, tuple[float, int]] = {}

    def allow(self, client: str) -> bool:
        """Record a request from ``client`` and return whether it is allowed."""
        now = time.monotonic()
        start, count = self._hits.get(client, (now, 0))
        if now - start >= self._window:
            start, count = now, 0
        count += 1
        self._hits[client] = (start, count)
        return count <= self._limit


def create_app(
    engine: Engine,
    *,
    cache_size: int = DEFAULT_CACHE_SIZE,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window: float = DEFAULT_RATE_WINDOW,
    cors_origins: Sequence[str] | None = None,
    frontend_url: str | None = None,
) -> FastAPI:
    """Build the FastAPI app around a RAG engine.

    Args:
        engine: The RAG engine to serve.
        cache_size: Max number of distinct answers to cache.
        rate_limit: Max requests per client per ``rate_window``.
        rate_window: Rate-limit window in seconds.
        cors_origins: Browser origins allowed to call the API cross-origin (the
            deployed frontend URL). When empty/omitted, no CORS middleware is
            added and the API is same-origin only.
        frontend_url: When set, the API root (``/``) redirects browsers here —
            the deployed single-page UI. When omitted there is no root route and
            the API is JSON-only (``/docs`` still serves the OpenAPI explorer).

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(
        title="BOE RAG Assistant",
        version="0.1.0",
        description="Answers about Spanish legislation with verifiable BOE citations.",
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    cache = _AnswerCache(cache_size)
    limiter = _RateLimiter(rate_limit, rate_window)

    if frontend_url:

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse(url=frontend_url)

    @app.middleware("http")
    async def _rate_limit(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in _RATE_LIMITED_PATHS:
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                return JSONResponse(
                    status_code=429, content={"detail": "rate limit exceeded"}
                )
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", num_chunks=engine.num_chunks)

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        results = engine.search(request.query, request.k)
        return SearchResponse(query=request.query, results=results)

    @app.post("/ask", response_model=AnswerResponse)
    def ask(request: AskRequest) -> AnswerResponse:
        key = (request.question, request.k)
        cached = cache.get(key)
        if cached is not None:
            return cached
        try:
            response = engine.answer(request.question, request.k)
        except LLMError as err:
            logger.warning("Answer generation failed: %s", err)
            raise HTTPException(
                status_code=503, detail="answer generation is unavailable"
            ) from err
        cache.put(key, response)
        return response

    return app
