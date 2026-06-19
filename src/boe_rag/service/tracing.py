"""Tracing seam for the query pipeline.

A minimal, dependency-free observability interface so each pipeline stage
(retrieve, rerank, generate) can later be exported to an external tracer such as
Langfuse (plan Phase 6) without changing the engine. The default
:class:`NoOpTracer` does nothing, so the engine behaves identically whether or
not an observability backend is wired in — keeping the heavy dependency and the
network egress out of CI and out of the default serving path.

The :class:`LangfuseTracer` adapter implements :class:`Tracer` by opening a
Langfuse span in :meth:`Tracer.span` and forwarding :meth:`Span.update` to it;
nothing else in the codebase changes. :func:`build_tracer` returns it when the
``LANGFUSE_*`` environment variables are set and a :class:`NoOpTracer` otherwise,
so observability is opt-in and the ``langfuse`` package stays an optional extra.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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


class _LangfuseSpanHandle(Protocol):
    """The slice of a Langfuse span object this adapter calls."""

    def update(self, **kwargs: object) -> None:
        """Forward keyword fields (``output``, ``metadata``, ...) to Langfuse."""
        ...


class _LangfuseClient(Protocol):
    """The slice of the Langfuse client this adapter calls.

    Targets the OpenTelemetry-based Langfuse v3 SDK, whose
    ``start_as_current_span`` is a context manager and whose nested calls
    automatically parent to the currently active span — giving the
    answer → retrieve → rerank → generate tree for free.
    """

    def start_as_current_span(
        self, *, name: str, input: object = None
    ) -> AbstractContextManager[_LangfuseSpanHandle]:
        """Open a Langfuse span as the current span and return it."""
        ...


class _LangfuseSpan:
    """Adapts a Langfuse span to the :class:`Span` protocol."""

    def __init__(self, handle: _LangfuseSpanHandle) -> None:
        """Wrap the underlying Langfuse span handle."""
        self._handle = handle

    def update(
        self,
        *,
        output: object = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Forward the provided fields to the Langfuse span."""
        fields: dict[str, object] = {}
        if output is not None:
            fields["output"] = output
        if metadata is not None:
            fields["metadata"] = metadata
        if fields:
            self._handle.update(**fields)


class LangfuseTracer:
    """A :class:`Tracer` backed by Langfuse.

    Each pipeline stage becomes a Langfuse span; nested stages nest in the trace
    because Langfuse parents new spans to the currently active one. The client is
    injected (built by :func:`build_tracer`) so this adapter carries no hard
    dependency on the ``langfuse`` package and is testable with a fake.

    Args:
        client: A Langfuse client (anything matching :class:`_LangfuseClient`).
    """

    def __init__(self, client: _LangfuseClient) -> None:
        """Bind the Langfuse client used to open spans."""
        self._client = client

    @contextmanager
    def span(self, name: str, **inputs: object) -> Iterator[Span]:
        """Open a Langfuse span for a pipeline stage, recording its inputs."""
        with self._client.start_as_current_span(
            name=name, input=dict(inputs)
        ) as handle:
            yield _LangfuseSpan(handle)


def build_tracer() -> Tracer:
    """Return a configured tracer for the serving pipeline.

    Returns a :class:`LangfuseTracer` when both ``LANGFUSE_PUBLIC_KEY`` and
    ``LANGFUSE_SECRET_KEY`` are set (the Langfuse client also reads
    ``LANGFUSE_HOST`` from the environment); otherwise returns a
    :class:`NoOpTracer`, so tracing stays opt-in and the ``langfuse`` package is
    imported only when it is actually wanted.

    Returns:
        A :class:`Tracer`: Langfuse-backed when configured, else a no-op.
    """
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return NoOpTracer()
    try:
        from langfuse import Langfuse  # optional dependency; imported lazily
    except ImportError:
        logger.warning(
            "LANGFUSE_* keys are set but the 'langfuse' package is not installed; "
            "tracing is disabled. Install the 'obs' extra to enable it."
        )
        return NoOpTracer()
    logger.info("Langfuse keys detected; enabling tracing.")
    return LangfuseTracer(Langfuse())
