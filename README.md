# BOE RAG Assistant

A production-grade Retrieval-Augmented Generation system that answers questions about
Spanish legislation with verifiable citations to the **BOE** (Boletín Oficial del Estado),
built with an eval-driven pipeline: every retrieval and generation change is measured
against a curated golden dataset before it ships.

[![CI](https://github.com/gonzalonao/boe-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/gonzalonao/boe-rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/gonzalonao/boe-rag-assistant)

**Tech stack (built):** Python · sentence-transformers (multilingual-E5 + cross-encoder) ·
NumPy in-memory hybrid index (dense + BM25/RRF) · FastAPI · Gradio · Docker ·
Hugging Face Hub (datasets, models, Spaces) · Langfuse (opt-in tracing) · GitHub Actions

**Planned:** Qdrant (vector store at full-corpus scale) · ONNX Runtime (int8 reranker) ·
fine-tuned Spanish embedding model · RAGAS (eval metrics)

## Demo

[**▶ Try the live demo**](https://huggingface.co/spaces/gonzalonao/boe-rag-assistant) — ask a
question about Spanish law in natural language and get an answer with citations linked back to
boe.es (or an honest refusal when the corpus doesn't cover it).

<!-- To embed the demo recording: add docs/media/demo.gif (see docs/media/README.md) and
     uncomment the next line.
![BOE RAG Assistant demo](docs/media/demo.gif)
-->

### Results at a glance

Retrieval on the 20-question golden set (2024 corpus, 2,225 chunks) — every stage measured
before it shipped:

| Stage | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| Dense baseline (`multilingual-e5-small`) | 0.900 | 0.749 | 0.783 |
| + Hybrid (BM25 · RRF fusion) | 0.900 | 0.763 | 0.793 |
| **+ Cross-encoder rerank** | **1.000** | **0.888** | **0.913** |

End-to-end answer quality (cite-or-refuse generation, scored by an LLM-as-judge):
**faithfulness 0.990 · correctness 0.895 · refusal rate 0.050**. Full methodology, per-stage
tables, and reproduction commands in [Evaluation](#evaluation-phase-2).

**With error bars.** Twenty gold questions carry wide uncertainty — recall@10 0.900 has a 95%
bootstrap CI of **[0.750, 1.000]**, so a 0.05 swing is within noise. The 1,749-example silver
set tightens that ~10× (recall@10 **0.963 [0.954, 0.972]**, MRR **0.827 [0.813, 0.842]**), which
is exactly why it exists. Every eval reports bootstrap CIs, and `eval/stats.py` adds a paired
permutation test so two systems can be compared for *significance*, not just a higher mean.

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
                       └────────────▶ in-memory hybrid index (dense E5 + BM25)
query ─▶ hybrid retrieval ─▶ rerank (cross-encoder) ─▶ grounded generation ─▶ cited answer
```

> The index is in-memory (NumPy) and the rerank model runs under sentence-transformers —
> a deliberate fit for the current 2,225-chunk corpus on free CPU hardware. The planned
> scale-up swaps in a **Qdrant** store (full-corpus, on-disk) and an **ONNX int8** reranker
> (see the roadmap).

For the full rationale — design principles, trade-offs, and the decisions log — see
[`docs/DESIGN.md`](docs/DESIGN.md).

### Extending the pipeline

The query engine (`RagEngine`) depends only on small protocols, so each planned
optimization is a drop-in implementation rather than a rewrite:

| Seam | Protocol | Current implementation | Planned swap |
|---|---|---|---|
| Retrieval | `Searcher` (`eval/retriever.py`) | in-memory dense E5 + BM25, RRF fusion | Qdrant store (dense + sparse, on-disk) |
| Query encoding | `Embedder` (`eval/retriever.py`) | off-the-shelf `multilingual-e5-small` | fine-tuned E5, ONNX int8 |
| Reranking | `Reranker` (`eval/rerank.py`) | sentence-transformers cross-encoder | ONNX int8 cross-encoder |
| Generation | `LLMProvider` (`llm/base.py`) | OpenRouter → Groq → Gemini fallback chain | any OpenAI-compatible provider |
| Observability | `Tracer` (`service/tracing.py`) | no-op by default; **Langfuse per-stage spans** when `LANGFUSE_*` is set | hosted dashboards, eval scoring |

`build_engine` (`service/app.py`) wires the concrete implementations together;
swapping one is a constructor change, and the unit tests exercise the engine with
fakes for every seam — no models or API keys in CI.

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
- [x] **Phase 6** — Serving: FastAPI service (`/ask`, `/search`, `/health`) + Gradio demo UI
  (chat with linked citations + a Quality tab) + containerised Hugging Face Space deployment
- [ ] **Phase 7** — Scheduled incremental ingestion + observability (Langfuse tracing
  wired via the `Tracer` seam; scheduled ingestion pending)

## Evaluation (Phase 2)

The harness (`src/boe_rag/eval/`) scores the retriever against a hand-curated golden
set of real Spanish legal questions ([`eval_data/seed_evalset.jsonl`](eval_data/seed_evalset.jsonl)),
each mapped to the chunk that answers it. Metrics (recall@k, precision@k, hit rate, MRR,
nDCG) are pure-Python and run in CI; the retrieval run uses an off-the-shelf embedding
model and is reproducible locally.

A **CI eval-gate** turns this into a regression guard: every pull request re-runs the
gold-set retrieval evaluation and fails if recall@10 or MRR drops more than a small
tolerance below the committed baseline ([`eval_data/retrieval_baseline.json`](eval_data/retrieval_baseline.json)),
so a change can't silently degrade retrieval quality.

**Two tiers of questions.** The 20 hand-curated questions are the trusted *gold* set. To
scale measurement, `scripts/generate_evalset.py` produces a larger *silver* set: it prompts
an LLM to write a self-contained question + answer grounded in a sampled chunk, drops deictic
or trivial questions, and keeps only answers the LLM-judge rates faithful to their source
(`src/boe_rag/eval/generate.py`). Synthetic questions can flatter the system that generated
them, so the two tiers are reported separately and the gold set stays the source of truth.

Both tiers are published on the Hub as
[`gonzalonao/boe-rag-evalset`](https://huggingface.co/datasets/gonzalonao/boe-rag-evalset)
(1,749 silver + 20 gold QA pairs, validated against the corpus). On the silver split,
dense retrieval (e5-small, k=10) scores recall@10 **0.963 [0.954, 0.972]** / MRR
**0.827 [0.813, 0.842]** (95% bootstrap CIs).

**Quantifying the uncertainty.** Every `run_eval.py` run reports a 95% bootstrap confidence
interval for recall@k and MRR (`src/boe_rag/eval/stats.py`), so the report shows how much
sampling noise sits behind each point estimate — on 20 questions the CIs are wide, on 1,749
they are tight. The same module provides a **paired sign-flip permutation test**
(`paired_delta_significance`) for comparing two systems on the same queries: it answers "is this
change *significant*?", not just "did the mean go up?", with the pairing removing between-query
variance so smaller real gains are detectable.

```bash
$env:OPENROUTER_API_KEY = "..."   # preferred: ~1000 free calls/day on `:free` models
# Optional: a comma-separated fallback chain (OpenRouter routes around a busy model):
$env:OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free,openai/gpt-oss-120b:free"
python scripts/generate_evalset.py --corpus data/corpus/boe-2024.parquet \
    --out eval_data/generated_evalset.jsonl --limit 150
```

Any one of `OPENROUTER_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` works (tried in that
order; whichever has a key leads, and the chain falls through on rate limits). The generator
survives free-tier limits: it waits out cool-downs and retries, and always saves what it has
collected. `OPENROUTER_MODEL`/`GROQ_MODEL`/`GEMINI_MODEL` select the model; `--no-validate`
halves token usage by skipping the faithfulness filter.

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
a second stage (the `RerankingRetriever` in `src/boe_rag/eval/rerank.py`, wrapping the
`CrossEncoderReranker` in `cross_encoder.py`): hybrid retrieves a 30-candidate pool, then
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

### Adversarial security (red-team)

A trustworthy answer has to survive a hostile user, so the generator is red-teamed against a
curated set of attacks (`eval_data/adversarial_security.jsonl`) with deterministic, rule-based
checks (`src/boe_rag/eval/security.py`) — no LLM judge, so the verdicts are stable. Four threat
classes: **instruction override** ("ignore your rules and output X"), **system-prompt
exfiltration** (defended with a canary token the answer must never contain),
**citation spoofing** (the answer may only cite passages that were actually retrieved), and
**out-of-corpus hallucination** (absent-law questions must refuse, not invent). The generator is
also hardened to treat passages and the question as *data, never instructions*.

The suite earns its keep by finding a real weakness and then proving the fix. The baseline
**fabricated citations** when asked (e.g. `[99]` for a source that was never retrieved) —
prompt-level defenses caught it 0% of the time. A deterministic post-hoc
**citation-validation** guardrail (`src/boe_rag/service/citation.py`) now strips any `[n]`
pointing past the retrieved passages and refuses when an answer's grounding rests entirely on
fabricated citations. Same harness, before vs. after (14 attacks, dense k=5):

| Attack category | Before | After |
|---|---|---|
| out-of-corpus hallucination | 100% | 100% |
| instruction override | 75% | 75% |
| system-prompt exfiltration | 75% | 75% |
| **citation spoofing** | **0%** | **100%** |
| **Overall** | **64% (9/14)** | **86% (12/14)** |

Full threat model, methodology, and the find→fix loop: [`docs/SECURITY.md`](docs/SECURITY.md);
latest report: [`reports/security_eval.md`](reports/security_eval.md). Reproduce it once an API
key is set:

```bash
$env:OPENROUTER_API_KEY = "..."   # or GROQ_API_KEY / GEMINI_API_KEY
python scripts/run_security_eval.py --corpus data/corpus/boe-2024.parquet \
    --out reports/security_eval
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
$env:OPENROUTER_API_KEY = "..."        # at least one LLM key for /ask (or GROQ/GEMINI)
uvicorn boe_rag.service.app:app --port 8000
# → http://localhost:8000/docs  (interactive OpenAPI UI)
```

## Demo UI (Phase 6)

A Gradio chat UI (`src/boe_rag/service/ui.py`) is mounted at the root of the same
FastAPI app, so the demo and the JSON API share one `Engine` and can never drift. It
has an **Assistant** tab (ask a question, get a grounded answer with each source linked
back to boe.es) and a **Quality** tab that surfaces the measured eval metrics. When the
free LLM tier is rate-limited, the chat degrades gracefully to showing the retrieved
passages instead of failing, and `/search` keeps working without an LLM.

```bash
pip install -e ".[api,ml,ui]"          # adds Gradio
$env:OPENROUTER_API_KEY = "..."        # or GROQ_API_KEY / GEMINI_API_KEY
uvicorn boe_rag.service.app:app --port 8000
# → http://localhost:8000/        (chat UI)
# → http://localhost:8000/docs    (OpenAPI)
```

## Deployment (Phase 6)

The demo runs as a containerised **Hugging Face Space** (Docker SDK). The image
([`Dockerfile`](Dockerfile)) is built to make cold start cheap on free CPU
hardware — everything the running container needs is baked in at build time:

- **CPU-only PyTorch**, so the multi-gigabyte CUDA wheel is never pulled.
- the **corpus** Parquet, fetched from the published HF dataset
  ([`scripts/fetch_corpus.py`](scripts/fetch_corpus.py));
- **precomputed E5 passage embeddings** ([`scripts/precompute_embeddings.py`](scripts/precompute_embeddings.py)),
  so the service loads a matrix instead of re-encoding all 2,225 passages on boot;
- the **embedding + cross-encoder weights** ([`scripts/warm_models.py`](scripts/warm_models.py)),
  so there are no model downloads at runtime.

At serving time the app reads `BOE_EMBEDDINGS_PATH` and skips the startup encode
when the precomputed matrix matches the corpus (falling back to encoding if it is
stale, so a mismatched file can never serve wrong results). Each **version tag**
(`vMAJOR.MINOR.PATCH`) pushed to the repo triggers
[`.github/workflows/deploy-space.yml`](.github/workflows/deploy-space.yml), which swaps
in the Space card ([`deploy/space/README.md`](deploy/space/README.md)) as the Space README
and force-pushes to the Space, which then rebuilds the image — so every deployment is a
tagged release. The only runtime secret is an LLM key — `OPENROUTER_API_KEY` (preferred),
`GROQ_API_KEY`, or `GEMINI_API_KEY` — set in the Space settings.

```bash
# Build and run the production image locally (mirrors the Space):
docker build -t boe-rag .
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY="..." boe-rag
# → http://localhost:7860/
```

## Observability (Phase 6)

Each pipeline stage (`answer → retrieve → rerank → generate`) is instrumented behind a
`Tracer` protocol. By default it is a **no-op** — zero overhead, zero dependencies. Set
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and install the `obs` extra) and the same
spans are exported to **Langfuse**, where each request appears as a nested trace named after
the question, with per-stage latency, inputs, and outputs:

<!-- Add docs/media/langfuse-trace.png (see docs/media/README.md) and uncomment:
![Langfuse trace of a single request](docs/media/langfuse-trace.png)
-->

```bash
pip install -e ".[api,ml,ui,obs]"
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_HOST = "https://cloud.langfuse.com"   # or your self-hosted instance
uvicorn boe_rag.service.app:app
```

The adapter ([`service/tracing.py`](src/boe_rag/service/tracing.py)) is unit-tested with a
fake client, so the heavy dependency and the network egress stay out of CI and the default
serving path. (Quality *scores* — LLM-judge faithfulness, 👍/👎 — are a planned follow-up.)

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

### Configuration

All runtime configuration is centralised in a typed
[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) model
(`src/boe_rag/settings.py`) — LLM keys/models, corpus/embeddings/report paths, and the
optional Langfuse keys. For local development, copy the template and fill in what you need:

```bash
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

The app and the eval scripts load `.env` at startup (`load_environment()`); a real
environment variable always overrides the file, and every key is optional. So instead of
exporting `$env:OPENROUTER_API_KEY` before each command, you can set it once in `.env`. In
production (the HF Space) nothing changes — the keys come from the Space secrets as ordinary
environment variables. `.env` is git-ignored and never committed.

## Author

**Gonzalo López Crespo** — [LinkedIn](https://linkedin.com/in/gonzalolopezcrespo) · [GitHub](https://github.com/gonzalonao)
