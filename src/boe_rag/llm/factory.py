"""Provider construction and a fallback chain.

``FallbackProvider`` tries each wrapped provider in order, moving on when one
fails (e.g. a free-tier rate limit), which keeps evals running across providers.
``build_available_providers`` constructs every provider whose API key is present.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider, LLMRateLimitError
from boe_rag.llm.gemini import GeminiProvider
from boe_rag.llm.groq import GroqProvider
from boe_rag.llm.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)

#: Default seconds to skip a provider after it rate-limits before retrying it.
DEFAULT_COOLDOWN_SECONDS = 60.0

#: Provider builders keyed by short name.
_BUILDERS: dict[str, Callable[[], LLMProvider]] = {
    "openrouter": OpenRouterProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def build_provider(name: str) -> LLMProvider:
    """Construct a single provider by short name (``openrouter``/``gemini``/``groq``).

    Args:
        name: Provider short name.

    Returns:
        The constructed provider.

    Raises:
        LLMError: If the name is unknown or the provider cannot be configured.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        raise LLMError(f"unknown provider {name!r}; choose from {sorted(_BUILDERS)}")
    return builder()


def build_available_providers(
    order: Sequence[str] = ("openrouter", "gemini", "groq"),
) -> list[LLMProvider]:
    """Build every provider in ``order`` whose API key is configured.

    Args:
        order: Provider short names in preference order.

    Returns:
        The providers that could be constructed (those with a key present).
    """
    providers: list[LLMProvider] = []
    for name in order:
        try:
            providers.append(build_provider(name))
        except LLMError as err:
            logger.info("Skipping provider %s: %s", name, err)
    return providers


class FallbackProvider:
    """Tries each wrapped provider in order until one succeeds.

    A rate-limited provider trips a time-based circuit breaker: it is skipped
    for a cool-down window rather than re-tried on every call, but recovers
    afterwards so a momentary free-tier burst doesn't disable it for the whole
    run. When every provider is cooling down, :class:`LLMRateLimitError` is
    raised (not a generic error) so a long-running caller can back off and
    retry instead of treating it as a permanent failure.

    Args:
        providers: Providers in preference order (at least one).
        cooldown: Seconds to skip a provider after it rate-limits.

    Raises:
        LLMError: If the provider list is empty.
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        cooldown: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        """Store the ordered provider chain and cool-down window."""
        if not providers:
            raise LLMError("FallbackProvider needs at least one provider")
        self._providers = list(providers)
        self._cooldown = cooldown
        #: Provider name -> monotonic time when it may be tried again.
        self._ready_at: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Identifier listing the chained providers."""
        return "fallback(" + ",".join(p.name for p in self._providers) + ")"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Try each provider not in cool-down, returning the first success.

        Raises:
            LLMRateLimitError: If every provider is rate-limited / cooling down.
            LLMError: If a provider fails for a non-rate-limit reason.
        """
        now = time.monotonic()
        errors: list[str] = []
        rate_limited = False
        other_error = False
        for provider in self._providers:
            ready_at = self._ready_at.get(provider.name)
            if ready_at is not None and now < ready_at:
                rate_limited = True
                continue
            try:
                result = provider.complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                self._ready_at.pop(provider.name, None)
                return result
            except LLMRateLimitError as err:
                self._ready_at[provider.name] = time.monotonic() + self._cooldown
                logger.warning(
                    "Provider %s rate-limited; cooling down for %.0fs.",
                    provider.name,
                    self._cooldown,
                )
                rate_limited = True
                errors.append(f"{provider.name}: {err}")
            except LLMError as err:
                logger.warning("Provider %s failed: %s", provider.name, err)
                other_error = True
                errors.append(f"{provider.name}: {err}")
        if rate_limited and not other_error:
            detail = " | ".join(errors) if errors else "all cooling down"
            raise LLMRateLimitError("all providers rate-limited -> " + detail)
        raise LLMError("all providers failed -> " + " | ".join(errors))
