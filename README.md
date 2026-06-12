# BOE RAG Assistant

A production-grade Retrieval-Augmented Generation system that answers questions about
Spanish legislation with verifiable citations to the **BOE** (Boletín Oficial del Estado),
built with an eval-driven pipeline: every retrieval and generation change is measured
against a curated golden dataset before it ships.

[![CI](https://github.com/gonzalonao/boe-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/gonzalonao/boe-rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Tech stack:** Python · Qdrant (hybrid dense + BM25 retrieval) · sentence-transformers ·
ONNX Runtime · FastAPI · Gradio · Hugging Face Hub (datasets, models, Spaces) · RAGAS ·
Langfuse · GitHub Actions

## Why this project

Most RAG demos are English-only toys with no measurement story. This one targets a real,
daily-updated Spanish legal corpus and treats retrieval quality as an engineering problem:

- **Eval-first:** golden QA dataset and retrieval metrics (recall@k, MRR, nDCG) exist
  *before* any optimization; CI blocks merges that regress them.
- **Measured improvements:** hybrid retrieval, cross-encoder reranking, and a fine-tuned
  Spanish embedding model — each shipped with a before/after benchmark table.
- **Grounded answers:** every claim cites the exact article of the law it comes from,
  with links back to boe.es.
- **Stays current:** scheduled ingestion keeps the index up to date with new BOE issues.

## Architecture

> Diagram coming with Phase 1 — see the roadmap below.

```
BOE Open Data API → ingestion → structure-aware chunking → HF dataset
                  → hybrid index (Qdrant: dense + BM25)
query → hybrid retrieval → rerank (ONNX cross-encoder) → grounded generation → cited answer
```

## Project status / roadmap

- [x] **Phase 0** — Scaffolding: tooling, CI, strict typing
- [ ] **Phase 1** — BOE ingestion pipeline → corpus dataset on HF Hub
- [ ] **Phase 2** — Eval harness + golden dataset + baseline RAG
- [ ] **Phase 3** — Retrieval engineering (hybrid search, reranking, chunking ablations)
- [ ] **Phase 4** — Embedding model fine-tune → published on HF Hub
- [ ] **Phase 5** — Grounded generation with citation validation
- [ ] **Phase 6** — FastAPI service + Gradio UI on a Hugging Face Space
- [ ] **Phase 7** — Scheduled incremental ingestion + observability

## Getting started

```bash
git clone https://github.com/gonzalonao/boe-rag-assistant.git
cd boe-rag-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

Run the quality suite (same checks as CI):

```bash
ruff format --check . && ruff check . && mypy && pytest
```

## Author

**Gonzalo López Crespo** — [LinkedIn](https://linkedin.com/in/gonzalolopezcrespo) · [GitHub](https://github.com/gonzalonao)
