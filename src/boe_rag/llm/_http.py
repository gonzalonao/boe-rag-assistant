"""Shared retrying JSON-POST helper for HTTP-based LLM providers.

Centralises timeout, retry-with-backoff on transient failures, and uniform
error mapping so each provider only has to build its request and parse its
response. Rate limits (HTTP 429) are surfaced as :class:`LLMRateLimitError` and
honour a ``Retry-After`` header when the server sends one.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from boe_rag.llm.base import LLMError, LLMRateLimitError

logger = logging.getLogger(__name__)

#: Status codes worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
#: The rate-limit status, handled specially so the fallback chain can react.
_RATE_LIMIT_STATUS = 429
#: Cap on how long we honour a server's ``Retry-After`` before giving up.
_MAX_RETRY_AFTER_SECONDS = 30.0

_FALLBACK_WAIT = wait_exponential(multiplier=0.5, max=8.0)


class _RetryableHTTPError(RuntimeError):
    """Internal signal that a response should trigger a retry.

    Attributes:
        status_code: The HTTP status that triggered the retry.
        retry_after: Server-requested delay in seconds, if any.
    """

    def __init__(self, status_code: int, retry_after: float | None) -> None:
        """Capture the status code and any ``Retry-After`` hint."""
        super().__init__(f"returned {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value given in seconds.

    Args:
        value: The raw header value, or ``None`` if absent.

    Returns:
        The delay in seconds, or ``None`` if absent or not a plain number.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _wait(retry_state: RetryCallState) -> float:
    """Wait for the server-requested delay if given, else exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableHTTPError) and exc.retry_after is not None:
        return min(exc.retry_after, _MAX_RETRY_AFTER_SECONDS)
    return _FALLBACK_WAIT(retry_state)


def post_json_with_retry(
    client: httpx.Client,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int = 4,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST JSON and return the parsed response, retrying transient failures.

    Args:
        client: The HTTP client to use.
        url: Target URL.
        json: JSON request body.
        headers: Request headers.
        max_retries: Maximum attempts before giving up.
        timeout: Per-request timeout in seconds.

    Returns:
        The parsed JSON response body.

    Raises:
        LLMRateLimitError: If the server keeps returning HTTP 429.
        LLMError: On any other non-retryable error or after exhausting retries.
    """

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, _RetryableHTTPError)),
        stop=stop_after_attempt(max_retries),
        wait=_wait,
        reraise=True,
    )
    def _call() -> dict[str, Any]:
        response = client.post(url, json=json, headers=headers, timeout=timeout)
        if response.status_code in _RETRYABLE_STATUS:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            raise _RetryableHTTPError(response.status_code, retry_after)
        if response.status_code >= 400:
            raise LLMError(
                f"{url} returned {response.status_code}: {response.text[:200]}"
            )
        parsed: dict[str, Any] = response.json()
        return parsed

    try:
        return _call()
    except _RetryableHTTPError as err:
        if err.status_code == _RATE_LIMIT_STATUS:
            raise LLMRateLimitError(f"{url} rate-limited (HTTP 429)") from err
        raise LLMError(f"{url} returned {err.status_code}") from err
    except httpx.HTTPError as err:
        raise LLMError(f"request to {url} failed: {err}") from err
