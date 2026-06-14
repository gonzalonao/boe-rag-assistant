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

```
BOE Open Data API ─▶ ingestion ─▶ structure-aware chunking ─▶ HF dataset (corpus)
   (sumario+XML)       │ client·parser·chunker                       │
                       │                                             ▼
                       └────────────▶ hybrid index (Qdrant: dense + BM25)
query ─▶ hybrid retrieval ─▶ rerank (ONNX cross-encoder) ─▶ grounded generation ─▶ cited answer
```

### Ingestion pipeline (Phase 1)

The ingestion layer (`src/boe_rag/ingest/`) turns the BOE Open Data API into a
clean, retrieval-ready corpus:

- **`client`** — resilient HTTP client: timeouts, polite rate limiting, and
  retry-with-backoff on transient failures.
- **`parser`** — flattens the daily *sumario* JSON into document references and
  parses each document's XML into typed metadata and ordered body blocks.
- **`chunker`** — *structure-aware*: walks the legal hierarchy
  (título → capítulo → sección → artículo) and emits one chunk per article with
  its full context, so every passage cites the exact article it came from.
- **`corpus`** — serialises chunks to Parquet for the Hugging Face Hub.

```bash
# Build a corpus for a date range
boe-ingest --start 2024-01-01 --end 2024-03-31 --out data/corpus/boe-2024-q1.parquet

# Publish it to the Hub (needs the `hub` extra and `huggingface-cli login`)
python scripts/push_corpus_to_hub.py \
    --parquet data/corpus/boe-2024-q1.parquet --repo-id <user>/boe-corpus
```

## Project status / roadmap

- [x] **Phase 0** — Scaffolding: tooling, CI, strict typing
- [x] **Phase 1** — BOE ingestion pipeline → corpus dataset on HF Hub
- [x] **Phase 2** — Eval harness: retrieval metrics + golden set + baseline, plus a
  provider-agnostic LLM layer (Gemini/Groq) and an LLM-as-judge end-to-end eval
  _(run the e2e eval once a free API key is set)_
- [ ] **Phase 3** — Retrieval engineering: ✅ hybrid BM25+dense (RRF); next: reranking, chunking ablations
- [ ] **Phase 4** — Embedding model fine-tune → published on HF Hub
- [ ] **Phase 5** — Grounded generation with citation validation
- [ ] **Phase 6** — FastAPI service + Gradio UI on a Hugging Face Space
- [ ] **Phase 7** — Scheduled incremental ingestion + observability

## Evaluation (Phase 2)

The harness (`src/boe_rag/eval/`) scores the retriever against a hand-curated golden
set of real Spanish legal questions ([`eval_data/seed_evalset.jsonl`](eval_data/seed_evalset.jsonl)),
each mapped to the chunk that answers it. Metrics (recall@k, precision@k, hit rate, MRR,
nDCG) are pure-Python and run in CI; the retrieval run uses an off-the-shelf embedding
model and is reproducible locally.

**Baseline** — `intfloat/multilingual-e5-small`, dense-only retrieval, 2024 corpus
(2,225 chunks, 20 questions), the "before" picture every later change is measured against:

| Recall@10 | Hit rate@10 | MRR | nDCG@10 |
|---|---|---|---|
| 0.900 | 0.900 | 0.749 | 0.783 |

Full report: [`reports/retrieval_baseline.md`](reports/retrieval_baseline.md). Reproduce it:

```bash
pip install -e ".[ml]"                 # embedding model (torch)
python scripts/run_eval.py --corpus data/corpus/boe-2024.parquet \
    --out reports/retrieval_baseline
```

### Hybrid retrieval (Phase 3)

Sparse and dense retrieval fail differently: BM25 nails exact legal references
(`artículo 14`, `Ley 39/2015`) but misses paraphrases; the embedding model matches
meaning but misses rare terms. A pure-Python BM25 index (`src/boe_rag/eval/sparse.py`)
is fused with the dense retriever by **Reciprocal Rank Fusion** (`hybrid.py`). Same corpus
and golden set as the baseline:

| Retriever | Recall@10 | Precision@10 | Hit rate@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| dense (baseline) | 0.900 | 0.090 | 0.900 | 0.749 | 0.783 |
| BM25 | 0.900 | 0.090 | 0.900 | 0.732 | 0.773 |
| **hybrid (RRF)** | **0.900** | **0.090** | **0.900** | **0.763** | **0.793** |

Hybrid is best on both ranking metrics (**MRR +0.014, nDCG +0.010** vs dense); recall holds
at the ceiling set by two hard questions both legs miss — a target for reranking and the
chunking ablation. Full report: [`reports/retrieval_hybrid.md`](reports/retrieval_hybrid.md).
Reproduce it:

```bash
python scripts/run_retrieval_ablation.py --corpus data/corpus/boe-2024.parquet \
    --out reports/retrieval_hybrid
```

### End-to-end (answer quality)

A provider-agnostic LLM layer (`src/boe_rag/llm/`, Gemini + Groq with a fallback chain)
powers both a baseline grounded answerer (cite-or-refuse prompting) and an **LLM-as-judge**
that scores each generated answer for **faithfulness** (grounded in the retrieved passages?)
and **correctness** (matches the reference answer?). Run it once an API key is set:

```bash
$env:GEMINI_API_KEY = "..."   # and/or $env:GROQ_API_KEY = "..."
python scripts/run_e2e_eval.py --corpus data/corpus/boe-2024.parquet \
    --out reports/e2e_baseline
```

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
