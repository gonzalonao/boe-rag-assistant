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
  provider-agnostic LLM layer (Gemini/Groq) and an LLM-as-judge end-to-end baseline
  (faithfulness 0.990, correctness 0.895)
- [x] **Phase 3** — Retrieval engineering: hybrid BM25+dense (RRF), cross-encoder reranking
  (recall 0.900→1.000), and a chunking ablation validating article-level chunks
- [ ] **Phase 4** — Embedding model fine-tune → published on HF Hub
- [ ] **Phase 5** — Grounded generation with citation validation
- [ ] **Phase 6** — Serving: ✅ FastAPI service (`/ask`, `/search`, `/health`); next: Gradio UI + HF Space
- [ ] **Phase 7** — Scheduled incremental ingestion + observability

## Evaluation (Phase 2)

The harness (`src/boe_rag/eval/`) scores the retriever against a hand-curated golden
set of real Spanish legal questions ([`eval_data/seed_evalset.jsonl`](eval_data/seed_evalset.jsonl)),
each mapped to the chunk that answers it. Metrics (recall@k, precision@k, hit rate, MRR,
nDCG) are pure-Python and run in CI; the retrieval run uses an off-the-shelf embedding
model and is reproducible locally.

**Two tiers of questions.** The 20 hand-curated questions are the trusted *gold* set. To
scale measurement, `scripts/generate_evalset.py` produces a larger *silver* set: it prompts
an LLM to write a self-contained question + answer grounded in a sampled chunk, drops deictic
or trivial questions, and keeps only answers the LLM-judge rates faithful to their source
(`src/boe_rag/eval/generate.py`). Synthetic questions can flatter the system that generated
them, so the two tiers are reported separately and the gold set stays the source of truth.

```bash
$env:GROQ_API_KEY = "..."   # Groq recommended; Gemini's free tier rate-limits hard
$env:GROQ_MODEL = "llama-3.1-8b-instant"   # bigger free daily-token budget for bulk jobs
python scripts/generate_evalset.py --corpus data/corpus/boe-2024.parquet \
    --out eval_data/generated_evalset.jsonl --limit 150
```

The generator survives free-tier limits: it waits out rate-limit cool-downs and retries,
and always saves what it has collected. `GROQ_MODEL`/`GEMINI_MODEL` select the model;
`--no-validate` halves token usage by skipping the faithfulness filter.

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

### Cross-encoder reranking (Phase 3)

A bi-encoder scores query and passage independently; a **cross-encoder** reads them together
and judges relevance directly — far sharper, but too slow for the whole corpus. So it runs as
a second stage (`src/boe_rag/eval/rerank.py`): hybrid retrieves a 30-candidate pool, then
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reorders it (~1.5 s/query on CPU).

| Retriever | Recall@10 | Precision@10 | Hit rate@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| dense (baseline) | 0.900 | 0.090 | 0.900 | 0.749 | 0.783 |
| hybrid (RRF) | 0.900 | 0.090 | 0.900 | 0.763 | 0.793 |
| **hybrid + cross-encoder** | **1.000** | **0.100** | **1.000** | **0.888** | **0.913** |

Reranking breaks the recall ceiling: **0.900 → 1.000** (the two hard questions were in the
candidate pool but ranked too low; the cross-encoder promotes them into the top 10), with
**MRR +0.125 and nDCG +0.120** over hybrid. Full report:
[`reports/retrieval_rerank.md`](reports/retrieval_rerank.md). Reproduce it:

```bash
python scripts/run_rerank_ablation.py --corpus data/corpus/boe-2024.parquet \
    --out reports/retrieval_rerank
```

### Chunking ablation (Phase 3)

Does structure-aware, article-level chunking actually beat the obvious alternatives? The
corpus is re-chunked three ways (`src/boe_rag/eval/chunking.py`) and scored at **document
granularity** (a hit = a chunk from a relevant document), so strategies with different chunk
boundaries are compared fairly:

| Strategy | Recall@10 | Hit rate@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **article (current)** | 1.000 | 1.000 | **0.975** | **0.982** |
| fixed-size (1000/150) | 1.000 | 1.000 | 0.912 | 0.935 |
| whole-document | 1.000 | 1.000 | 0.975 | 0.982 |

Every document is found (recall saturates on 105 docs / 20 questions), but ranking separates
them: **fixed-size windows rank the right document lower (MRR −0.063)** because they fragment
articles, while **article-level matches whole-document embedding — and uniquely supports
exact article citations** that whole-document loses. The production choice holds up. Full
report: [`reports/retrieval_chunking.md`](reports/retrieval_chunking.md). Reproduce it:

```bash
python scripts/run_chunking_ablation.py --corpus data/corpus/boe-2024.parquet \
    --out reports/retrieval_chunking
```

> Recall is saturated at this scale; finer signal needs the larger eval set (see roadmap).

### End-to-end (answer quality)

A provider-agnostic LLM layer (`src/boe_rag/llm/`, Gemini + Groq with a fallback chain
that trips a circuit breaker on a rate-limited provider) powers both a baseline grounded
answerer (cite-or-refuse prompting) and an **LLM-as-judge** that scores each generated
answer for **faithfulness** (grounded in the retrieved passages?) and **correctness**
(matches the reference answer?).

**Baseline** — dense retrieval (k=5) + cite-or-refuse generation, judged over the 20-question
golden set:

| Mean faithfulness | Mean correctness | Refusal rate |
|---|---|---|
| 0.990 | 0.895 | 0.050 |

Near-perfect faithfulness confirms the cite-or-refuse prompt rarely hallucinates; correctness
is the headroom that retrieval and generation work will target. Full report:
[`reports/e2e_baseline.md`](reports/e2e_baseline.md). Reproduce it once an API key is set:

```bash
$env:GEMINI_API_KEY = "..."   # and/or $env:GROQ_API_KEY = "..."
python scripts/run_e2e_eval.py --corpus data/corpus/boe-2024.parquet \
    --out reports/e2e_baseline
```

## API service (Phase 6)

The query pipeline is served by a FastAPI app (`src/boe_rag/service/`). A `RagEngine`
composes the measured pieces — hybrid retrieval → cross-encoder rerank → grounded,
cited generation — behind an `Engine` protocol, so the HTTP layer is unit-tested with a
fake engine (no models or API keys in CI). Endpoints:

- `GET /health` — readiness and indexed chunk count.
- `POST /search` — raw passage retrieval (`{"query": ..., "k": 10}`).
- `POST /ask` — grounded answer with citations (`{"question": ..., "k": 5}`); identical
  questions are cached and a per-client rate limit protects the free LLM tier.

```bash
pip install -e ".[api,ml]"             # service + embedding/rerank models
$env:GROQ_API_KEY = "..."              # at least one LLM key for /ask
uvicorn boe_rag.service.app:app --port 8000
# → http://localhost:8000/docs  (interactive OpenAPI UI)
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
