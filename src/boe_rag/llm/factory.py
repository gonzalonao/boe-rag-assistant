"""Provider construction and a fallback chain.

``FallbackProvider`` tries each wrapped provider in order, moving on when one
fails (e.g. a free-tier rate limit), which keeps evals running across providers.
``build_available_providers`` constructs every provider whose API key is present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from boe_rag.llm.base import ChatMessage, LLMError, LLMProvider
from boe_rag.llm.gemini import GeminiProvider
from boe_rag.llm.groq import GroqProvider

logger = logging.getLogger(__name__)

#: Provider builders keyed by short name.
_BUILDERS: dict[str, Callable[[], LLMProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def build_provider(name: str) -> LLMProvider:
    """Construct a single provider by short name (``gemini`` or ``groq``).

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
    order: Sequence[str] = ("gemini", "groq"),
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

    Args:
        providers: Providers in preference order (at least one).

    Raises:
        LLMError: If the provider list is empty.
    """

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        """Store the ordered provider chain."""
        if not providers:
            raise LLMError("FallbackProvider needs at least one provider")
        self._providers = list(providers)

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
        """Try each provider in turn, returning the first success.

        Raises:
            LLMError: If every provider fails.
        """
        errors: list[str] = []
        for provider in self._providers:
            try:
                return provider.complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            except LLMError as err:
                logger.warning("Provider %s failed: %s", provider.name, err)
                errors.append(f"{provider.name}: {err}")
        raise LLMError("all providers failed -> " + " | ".join(errors))
