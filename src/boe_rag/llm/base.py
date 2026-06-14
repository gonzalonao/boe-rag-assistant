"""Provider-agnostic LLM interface.

A minimal chat-completion contract so the rest of the system (baseline answer
generation, the LLM-as-judge) never depends on a specific vendor. Concrete
providers (Gemini, Groq) implement :class:`LLMProvider`; a fallback wrapper can
chain them to ride out free-tier rate limits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

#: Allowed chat roles, mapped to each provider's own vocabulary internally.
Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails or is misconfigured."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single chat message.

    Attributes:
        role: The speaker role.
        content: The message text.
    """

    role: Role
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """A chat-completion provider."""

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``gemini:gemini-2.0-flash``."""
        ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's completion for a chat message sequence.

        Args:
            messages: The conversation so far.
            temperature: Sampling temperature; 0 for deterministic judging.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text.

        Raises:
            LLMError: If the call fails after retries.
        """
        ...
