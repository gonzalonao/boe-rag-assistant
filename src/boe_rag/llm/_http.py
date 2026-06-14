"""Shared retrying JSON-POST helper for HTTP-based LLM providers.

Centralises timeout, retry-with-backoff on transient failures, and uniform
error mapping so each provider only has to build its request and parse its
response.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from boe_rag.llm.base import LLMError

logger = logging.getLogger(__name__)

#: Status codes worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class _RetryableHTTPError(RuntimeError):
    """Internal signal that a response should trigger a retry."""


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
        LLMError: On a non-retryable HTTP error or after exhausting retries.
    """

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, _RetryableHTTPError)),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=0.5, max=8.0),
        reraise=True,
    )
    def _call() -> dict[str, Any]:
        response = client.post(url, json=json, headers=headers, timeout=timeout)
        if response.status_code in _RETRYABLE_STATUS:
            raise _RetryableHTTPError(f"{url} returned {response.status_code}")
        if response.status_code >= 400:
            raise LLMError(
                f"{url} returned {response.status_code}: {response.text[:200]}"
            )
        parsed: dict[str, Any] = response.json()
        return parsed

    try:
        return _call()
    except _RetryableHTTPError as err:
        raise LLMError(str(err)) from err
    except httpx.HTTPError as err:
        raise LLMError(f"request to {url} failed: {err}") from err
