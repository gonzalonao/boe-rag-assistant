---
title: BOE RAG Assistant
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Eval-driven RAG over Spain's official state gazette (BOE).
---

# ⚖️ BOE RAG Assistant

Ask questions about Spanish legislation (BOE — *Boletín Oficial del Estado*) and
get answers grounded in the official text, with verifiable citations linking back
to boe.es.

This Space serves the **JSON API** (`/ask`, `/search`, `/health`, `/docs`) of an
eval-driven Retrieval-Augmented Generation pipeline. The user interface is a
custom React app — **opening this Space redirects you to it**:
**[gonzalonao.github.io/boe-rag-assistant](https://gonzalonao.github.io/boe-rag-assistant/)**.

1. **Hybrid retrieval** — sparse BM25 + dense embeddings fused with Reciprocal
   Rank Fusion.
2. **Cross-encoder reranking** — a second stage that lifted retrieval recall
   from 0.90 to 1.00 on the golden set.
3. **Grounded generation** — cite-or-refuse prompting, so the model answers only
   from the retrieved BOE passages or declines.

> Source code, full eval reports, and engineering notes:
> **[github.com/gonzalonao/boe-rag-assistant](https://github.com/gonzalonao/boe-rag-assistant)**
>
> Built by [Gonzalo López Crespo](https://linkedin.com/in/gonzalolopezcrespo).
