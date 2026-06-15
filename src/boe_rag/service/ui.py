"""Gradio demo UI for the RAG assistant, mounted on the FastAPI app.

The UI is a thin presentation layer over the same :class:`Engine` the JSON API
uses, so the demo and the API can never drift apart. Two tabs: a chat assistant
that shows each answer's grounding sources (linked back to boe.es), and a quality
tab that surfaces the project's measured eval metrics.

Gradio is a heavy, optional dependency (the ``ui`` extra) and is imported lazily
inside :func:`build_ui`, so this module's pure helpers — :func:`format_sources`
and :func:`render_quality_markdown` — stay importable and unit-testable without
it. When generation is rate-limited, the chat degrades to showing retrieved
passages instead of failing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from boe_rag.llm.base import LLMError
from boe_rag.service.engine import Engine
from boe_rag.service.models import Source

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import gradio as gr

#: Max characters of a passage shown under an answer before truncating.
_SNIPPET_CHARS = 320
#: Default number of passages to ground an answer in.
_DEFAULT_K = 5

_HEADER_MD = (
    "# ⚖️ BOE RAG Assistant\n"
    "Ask questions about **2024 Spanish legislation** (BOE — *Boletín Oficial "
    "del Estado*) and get answers grounded in the official text, with verifiable "
    "citations linking back to boe.es. Questions are best asked in Spanish."
)

_ABOUT_MD = (
    "### About this demo\n"
    "An eval-driven Retrieval-Augmented Generation pipeline over Spain's official "
    "gazette:\n\n"
    "1. **Hybrid retrieval** — sparse BM25 + dense embeddings fused with "
    "Reciprocal Rank Fusion.\n"
    "2. **Cross-encoder reranking** — a second stage that reorders candidates "
    "(lifted retrieval recall from 0.90 to 1.00 on the golden set).\n"
    "3. **Grounded generation** — cite-or-refuse prompting, so the model answers "
    "only from the retrieved BOE passages or declines.\n\n"
    "Every answer lists the passages it is grounded in, each linking to the "
    "source document on boe.es. See the **Quality** tab for measured metrics.\n\n"
    "Built by [Gonzalo López Crespo](https://linkedin.com/in/gonzalolopezcrespo) "
    "· [GitHub](https://github.com/gonzalonao/boe-rag-assistant)"
)

_EXAMPLE_QUESTIONS = [
    "¿Qué se entiende por subvención según la legislación española?",
    "¿Qué requisitos debe cumplir una entidad para recibir una subvención?",
    "¿Qué obligaciones tiene el beneficiario de una subvención?",
]

_RATE_LIMITED_NOTICE = (
    "⚠️ _Answer generation is rate-limited right now (free LLM tier). Here are "
    "the most relevant passages so you can still find the answer:_"
)


def format_sources(sources: Sequence[Source]) -> str:
    """Render retrieved passages as a Markdown list of linked citations.

    Args:
        sources: The passages to render, best first.

    Returns:
        A Markdown string (empty if there are no sources).
    """
    if not sources:
        return ""
    lines = ["", "**Sources**"]
    for index, source in enumerate(sources, start=1):
        snippet = source.text.strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"{index}. [{source.citation}]({source.url}) — {snippet}")
    return "\n".join(lines)


def _format_metric_row(label: str, metrics: Mapping[str, float]) -> str:
    """Render one retrieval configuration as a Markdown table row."""
    return (
        f"| {label} | {metrics['recall_at_k']:.3f} | {metrics['hit_rate_at_k']:.3f} "
        f"| {metrics['mrr']:.3f} | {metrics['ndcg_at_k']:.3f} |"
    )


def render_quality_markdown(
    retrieval: Mapping[str, Mapping[str, float]],
    e2e: Mapping[str, float],
) -> str:
    """Render the eval metrics as a Markdown report for the Quality tab.

    Args:
        retrieval: Map of configuration name → retrieval metrics (recall_at_k,
            hit_rate_at_k, mrr, ndcg_at_k), e.g. the retrieval-rerank report.
        e2e: End-to-end answer metrics (mean_faithfulness, mean_correctness,
            refusal_rate).

    Returns:
        A Markdown string; a fallback note if a section's data is missing.
    """
    parts: list[str] = ["### Measured quality"]

    if retrieval:
        sample = next(iter(retrieval.values()))
        k = int(sample.get("k", 10))
        n = int(sample.get("num_queries", 0))
        parts.append(
            f"**Retrieval** — golden set of {n} questions, evaluated at k={k}:"
        )
        parts.append("")
        parts.append("| Configuration | recall@k | hit@k | MRR | nDCG@k |")
        parts.append("|---|---|---|---|---|")
        parts.extend(
            _format_metric_row(label, metrics) for label, metrics in retrieval.items()
        )
    else:
        parts.append("_Retrieval metrics unavailable._")

    parts.append("")
    if e2e:
        parts.append("**End-to-end answer quality** (LLM-as-judge):")
        parts.append("")
        parts.append("| Mean faithfulness | Mean correctness | Refusal rate |")
        parts.append("|---|---|---|")
        parts.append(
            f"| {e2e.get('mean_faithfulness', 0.0):.3f} "
            f"| {e2e.get('mean_correctness', 0.0):.3f} "
            f"| {e2e.get('refusal_rate', 0.0):.3f} |"
        )
    else:
        parts.append("_End-to-end metrics unavailable._")

    return "\n".join(parts)


def build_ui(engine: Engine, quality_markdown: str) -> gr.Blocks:
    """Build the Gradio Blocks UI bound to a RAG engine.

    Args:
        engine: The RAG engine answering questions and retrieving passages.
        quality_markdown: Pre-rendered metrics for the Quality tab (see
            :func:`render_quality_markdown`).

    Returns:
        A Gradio ``Blocks`` app, ready to mount on the FastAPI application.
    """
    import gradio as gr

    def respond(message: str, history: list[Any], k: int) -> str:
        """Answer a question, degrading to retrieval if generation is limited."""
        question = message.strip()
        if not question:
            return "Please enter a question."
        try:
            result = engine.answer(question, int(k))
        except LLMError:
            passages = engine.search(question, int(k))
            return f"{_RATE_LIMITED_NOTICE}\n{format_sources(passages)}"
        if result.refused:
            return result.answer
        return f"{result.answer}\n{format_sources(result.sources)}"

    with gr.Blocks(title="BOE RAG Assistant", fill_height=True) as demo:
        gr.Markdown(_HEADER_MD)
        with gr.Tabs():
            with gr.Tab("Assistant"):
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=_DEFAULT_K,
                    step=1,
                    label="Passages to ground the answer in (k)",
                )
                gr.ChatInterface(
                    fn=respond,
                    type="messages",
                    additional_inputs=[k_slider],
                    examples=[[q] for q in _EXAMPLE_QUESTIONS],
                    cache_examples=False,
                )
            with gr.Tab("Quality"):
                gr.Markdown(quality_markdown)
            with gr.Tab("About"):
                gr.Markdown(_ABOUT_MD)

    return cast("gr.Blocks", demo)
