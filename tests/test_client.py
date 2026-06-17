"""Tests for the BOE HTTP client's retry and error handling."""

from __future__ import annotations

import httpx
import pytest

from boe_rag.config import IngestionConfig
from boe_rag.ingest.client import BoeApiError, BoeClient


def _client_with_handler(
    handler: object,
    *,
    max_retries: int = 4,
) -> BoeClient:
    """Build a BoeClient whose transport is driven by a mock handler."""
    config = IngestionConfig(min_request_interval_s=0.0, max_retries=max_retries)
    client = BoeClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return client


def test_fetch_document_retries_then_succeeds() -> None:
    """A transient 503 is retried until a 200 is returned."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text="<documento/>")

    with _client_with_handler(handler) as client:
        assert client.fetch_document_xml("BOE-A-2024-1") == "<documento/>"
    assert calls["n"] == 3


def test_fetch_document_gives_up_after_max_retries() -> None:
    """Persistent retryable errors surface as a BoeApiError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    with _client_with_handler(handler, max_retries=2) as client:  # noqa: SIM117
        with pytest.raises(BoeApiError, match="retryable status 503"):
            client.fetch_document_xml("BOE-A-2024-1")


def test_non_retryable_status_raises_immediately() -> None:
    """A 404 is not retried and raises a BoeApiError."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="not found")

    with _client_with_handler(handler) as client:  # noqa: SIM117
        with pytest.raises(BoeApiError):
            client.fetch_document_xml("BOE-A-2024-1")
    assert calls["n"] == 1


def test_fetch_sumario_rejects_non_ok_status() -> None:
    """A 200 response whose JSON status is not 200 is treated as an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": {"code": "404", "text": "no"}})

    with _client_with_handler(handler) as client:  # noqa: SIM117
        with pytest.raises(BoeApiError, match="status"):
            client.fetch_sumario("20240115")
