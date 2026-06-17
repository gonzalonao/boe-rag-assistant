"""HTTP client for the BOE open-data API.

Wraps the daily sumario and document-XML endpoints with timeouts, polite rate
limiting, and retry-with-backoff on transient failures, so the rest of the
pipeline can treat fetching as reliable.
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from boe_rag.config import (
    DOCUMENT_XML_URL,
    SUMARIO_API_URL,
    IngestionConfig,
)

logger = logging.getLogger(__name__)

#: HTTP status codes worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class BoeApiError(RuntimeError):
    """Raised when the BOE API returns a non-recoverable error response."""


class _RetryableResponseError(RuntimeError):
    """Internal signal that a response should trigger a retry."""


class BoeClient:
    """Polite, resilient client for the BOE open-data endpoints.

    Use as a context manager so the underlying connection pool is closed::

        with BoeClient() as client:
            payload = client.fetch_sumario("20240115")

    Args:
        config: Ingestion configuration controlling timeouts, retries, and the
            minimum interval between requests.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        """Initialise the client and its HTTP connection pool."""
        self._config = config or IngestionConfig()
        self._client = httpx.Client(
            timeout=self._config.request_timeout_s,
            headers={
                "User-Agent": "boe-rag-assistant/0.1 (+https://github.com/gonzalonao)"
            },
            follow_redirects=True,
        )
        self._last_request_ts = 0.0

    def __enter__(self) -> Self:
        """Enter the context manager, returning this client."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client on context exit."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def fetch_sumario(self, date_str: str) -> dict[str, Any]:
        """Fetch the daily sumario (index) for a given date.

        Args:
            date_str: Date in ``YYYYMMDD`` format.

        Returns:
            The parsed JSON payload of the sumario response.

        Raises:
            BoeApiError: If the API reports a non-OK status or no issue exists
                for the date.
        """
        url = f"{SUMARIO_API_URL}/{date_str}"
        response = self._request(url, accept="application/json")
        payload: dict[str, Any] = response.json()
        status = payload.get("status", {})
        if str(status.get("code")) != "200":
            raise BoeApiError(
                f"Sumario for {date_str} returned status {status!r}",
            )
        return payload

    def fetch_document_xml(self, identifier: str) -> str:
        """Fetch the raw XML of a single document.

        Args:
            identifier: BOE document id, e.g. ``BOE-A-2024-714``.

        Returns:
            The document XML as text.

        Raises:
            BoeApiError: If the document cannot be retrieved.
        """
        url = DOCUMENT_XML_URL.format(identifier=identifier)
        response = self._request(url, accept="application/xml")
        return response.text

    def _request(self, url: str, *, accept: str) -> httpx.Response:
        """Perform a rate-limited, retrying GET request."""
        try:
            return self._request_with_retry(url, accept)
        except _RetryableResponseError as err:
            raise BoeApiError(str(err)) from err
        except httpx.HTTPError as err:
            raise BoeApiError(f"Request to {url} failed: {err}") from err

    def _make_retrying_call(self) -> Any:
        """Build the tenacity-decorated request callable bound to this config."""

        @retry(
            retry=retry_if_exception_type(
                (httpx.TransportError, _RetryableResponseError)
            ),
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential(multiplier=0.5, max=8.0),
            reraise=True,
        )
        def _call(url: str, accept: str) -> httpx.Response:
            self._throttle()
            logger.debug("GET %s", url)
            response = self._client.get(url, headers={"Accept": accept})
            if response.status_code in _RETRYABLE_STATUS:
                raise _RetryableResponseError(
                    f"{url} returned retryable status {response.status_code}"
                )
            response.raise_for_status()
            return response

        return _call

    def _request_with_retry(self, url: str, accept: str) -> httpx.Response:
        """Execute the retrying request callable."""
        call = self._make_retrying_call()
        result: httpx.Response = call(url, accept)
        return result

    def _throttle(self) -> None:
        """Sleep just enough to honour the minimum inter-request interval."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._config.min_request_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()
