"""Tracing seam for the query pipeline.

A minimal, dependency-free observability interface so each pipeline stage
(retrieve, rerank, generate) can later be exported to an external tracer such as
Langfuse (plan Phase 6) without changing the engine. The default
:class:`NoOpTracer` does nothing, so the engine behaves identically whether or
not an observability backend is wired in — keeping the heavy dependency and the
network egress out of CI and out of the default serving path.

A future Langfuse adapter implements :class:`Tracer` by opening a Langfuse span
in :meth:`Tracer.span` and forwarding :meth:`Span.update` to it; nothing else in
the codebase needs to change.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class Span(Protocol):
    """A single timed unit of work within a trace."""

    def update(
        self,
        *,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Attach an output value and/or metadata to the span.

        Args:
            output: The stage's result, recorded for inspection (e.g. the answer
                text or the number of passages returned).
            metadata: Arbitrary structured detail (scores, pool sizes, flags).
        """
        ...


@runtime_checkable
class Tracer(Protocol):
    """Opens spans for pipeline stages; implemented by an observability adapter."""

    def span(self, name: str, **inputs: object) -> AbstractContextManager[Span]:
        """Open a span named ``name``.

        Args:
            name: Stage name (e.g. ``"retrieve"``, ``"rerank"``, ``"generate"``).
            **inputs: The stage's inputs, recorded on the span.

        Returns:
            A context manager yielding the :class:`Span` for the stage; the span
            is closed (and its duration recorded) on exit.
        """
        ...


class _NoOpSpan:
    """A span that records nothing."""

    def update(
        self,
        *,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Discard the update."""
        return None


class NoOpTracer:
    """Default tracer: opens spans that do nothing.

    Lets the engine instrument every stage unconditionally while keeping zero
    overhead and zero dependencies until a real backend is wired in.
    """

    @contextmanager
    def span(self, name: str, **inputs: object) -> Iterator[Span]:
        """Yield a no-op span and discard it on exit."""
        yield _NoOpSpan()
