"""Tests for the FastAPI service, using a fake engine (no models or API keys)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from boe_rag.llm.base import LLMError
from boe_rag.service.api import create_app
from boe_rag.service.models import AnswerResponse, Source


class _FakeEngine:
    """In-memory engine returning canned results and counting answer calls."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.answer_calls = 0

    @property
    def num_chunks(self) -> int:
        return 3

    def search(self, query: str, k: int = 10) -> list[Source]:
        source = Source(
            chunk_id="c1",
            citation="Ley 1/2024, Artículo 1",
            text="El tipo general del IVA es del 21%.",
            url="https://www.boe.es/diario_boe/txt.php?id=BOE-A-2024-1",
            score=1.0,
        )
        return [source][:k]

    def answer(self, query: str, k: int = 5) -> AnswerResponse:
        self.answer_calls += 1
        if self._fail:
            raise LLMError("provider down")
        return AnswerResponse(
            answer="El tipo general del IVA es del 21% [1].",
            refused=False,
            sources=self.search(query, k),
        )


def test_health_reports_chunk_count() -> None:
    """/health returns ok and the number of indexed chunks."""
    client = TestClient(create_app(_FakeEngine()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "num_chunks": 3}


def test_search_returns_sources() -> None:
    """/search echoes the query and returns retrieved passages."""
    client = TestClient(create_app(_FakeEngine()))
    response = client.post("/search", json={"query": "tipo de IVA", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "tipo de IVA"
    assert body["results"][0]["chunk_id"] == "c1"
    assert body["results"][0]["url"].startswith("https://www.boe.es")


def test_ask_returns_answer_and_caches_it() -> None:
    """/ask returns a grounded answer and caches identical requests."""
    engine = _FakeEngine()
    client = TestClient(create_app(engine))
    payload = {"question": "¿Cuál es el tipo general del IVA?", "k": 3}

    first = client.post("/ask", json=payload)
    assert first.status_code == 200
    assert "21%" in first.json()["answer"]
    assert first.json()["sources"][0]["chunk_id"] == "c1"

    second = client.post("/ask", json=payload)
    assert second.status_code == 200
    assert engine.answer_calls == 1  # served from cache the second time


def test_ask_rejects_empty_question() -> None:
    """A blank question fails request validation."""
    client = TestClient(create_app(_FakeEngine()))
    response = client.post("/ask", json={"question": "", "k": 3})
    assert response.status_code == 422


def test_ask_returns_503_when_llm_fails() -> None:
    """An LLM failure surfaces as a 503, not a crash."""
    client = TestClient(create_app(_FakeEngine(fail=True)))
    response = client.post("/ask", json={"question": "x y z", "k": 3})
    assert response.status_code == 503


def test_rate_limit_blocks_excess_requests() -> None:
    """Requests to an API endpoint beyond the per-client limit get a 429."""
    client = TestClient(create_app(_FakeEngine(), rate_limit=2))
    payload = {"query": "tipo de IVA", "k": 5}
    assert client.post("/search", json=payload).status_code == 200
    assert client.post("/search", json=payload).status_code == 200
    assert client.post("/search", json=payload).status_code == 429


def test_health_is_not_rate_limited() -> None:
    """/health is exempt so the UI and probes are never throttled."""
    client = TestClient(create_app(_FakeEngine(), rate_limit=1))
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200


def test_cors_allows_configured_origin() -> None:
    """A configured frontend origin gets an Access-Control-Allow-Origin header."""
    origin = "https://boe-rag.example.app"
    client = TestClient(create_app(_FakeEngine(), cors_origins=[origin]))
    response = client.post(
        "/search", json={"query": "IVA", "k": 5}, headers={"origin": origin}
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_absent_by_default() -> None:
    """With no configured origins the API stays same-origin (no CORS header)."""
    client = TestClient(create_app(_FakeEngine()))
    response = client.post(
        "/search",
        json={"query": "IVA", "k": 5},
        headers={"origin": "https://x.example"},
    )
    assert "access-control-allow-origin" not in response.headers
