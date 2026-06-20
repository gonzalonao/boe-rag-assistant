# Design

How the BOE RAG Assistant is built and *why* it is built that way. The README is the tour;
this is the rationale. For the narrative walk-through of the retrieval/eval engineering, see
the [technical deep-dive](https://gonzalonao.dev/writing/boe-rag-assistant-deep-dive); for the
running roadmap, see `RAG-ASSISTANT-PROJECT-PLAN.md`.

## 1. Problem

Spanish legal questions need answers that are **verifiable**. A confident paraphrase of the law
is worse than useless — a citizen or lawyer must be able to check the source. The corpus (the
**BOE**, *Boletín Oficial del Estado*) is large, Spanish, and grows daily, so retrieval quality
is a genuine engineering problem rather than a toy.

## 2. Goals and non-goals

**Goals**
- Every answer cites the exact article it used, linked back to boe.es — or refuses.
- Retrieval/generation quality is **measured**: no change ships without a before/after number
  on a fixed evaluation set, enforced in CI.
- The system runs on **free CPU hardware** (a Hugging Face Space) with a cheap cold start.
- The architecture is **swap-ready**: each expensive component sits behind a small protocol so
  it can be replaced (vector DB, ONNX reranker, fine-tuned embeddings) without a rewrite.

**Non-goals**
- Not legal advice; not authoritative — boe.es remains the source of truth.
- Not a low-latency production SLA; the demo prioritises correctness and cost over speed.
- No conversational memory — each question is answered independently (keeps grounding tight).

## 3. Design principles

1. **Eval-first.** The measurement harness exists before the optimisation. Quality is a CI gate
   (`boe_rag.eval.regression` + `.github/workflows/ci.yml`), not a vibe.
2. **Protocol seams.** `RagEngine` (`service/engine.py`) depends only on small `Protocol`s —
   `Searcher`, `Embedder`, `Reranker`, `LLMProvider`, `Tracer`. Concrete, heavy implementations
   are injected by `build_engine` (`service/app.py`). The logic is therefore testable with fakes
   — **no models or API keys in CI**.
3. **Honest scope.** The README separates the *built* stack from the *planned* one. Numbers are
   reported per tier (gold vs silver eval set) and never inflated.
4. **Cheap by construction.** In-memory NumPy index + pure-Python BM25 + precomputed embeddings
   keep the Space within free-tier limits; the heavier swaps are deferred until the corpus
   actually needs them.

## 4. Architecture

Two halves connected by a versioned corpus artifact:

```
OFFLINE (ingestion)                         ONLINE (query)
BOE Open Data API                           question
  │ client · parser · chunker                 │
  ▼                                           ▼
structure-aware chunks  ──▶ HF dataset ──▶ in-memory hybrid index (dense E5 + BM25)
  (one per article)        (Parquet)         │
                                             ▼  hybrid retrieval (RRF)
                                             ▼  cross-encoder rerank (top-k of a 30 pool)
                                             ▼  grounded generation (cite-or-refuse)
                                          cited answer + sources
```

### 4.1 Ingestion (`src/boe_rag/ingest/`)
- **`client`** — resilient HTTP over the BOE Open Data API: timeouts, polite rate limiting,
  retry-with-backoff.
- **`parser`** — daily *sumario* JSON → document refs; per-document XML → typed metadata +
  ordered body blocks.
- **`chunker`** — **structure-aware**: walks `título → capítulo → sección → artículo` and emits
  **one chunk per article**, carrying its hierarchy as metadata so each chunk can build its own
  citation (`Ley 39/2015, Artículo 3`). Oversized articles split into numbered parts only after
  a chunk holds a heading *plus* a paragraph (headings are never orphaned); sub-minimum
  fragments merge into the preceding chunk *within the same scope*.
- **`corpus`** — serialises to Parquet for the Hub (`gonzalonao/boe-corpus`).

Chunking is treated as a **retrieval decision, not preprocessing** — validated by an ablation
(article vs fixed-size vs whole-document), see §5.

### 4.2 Query pipeline (`src/boe_rag/eval/` + `service/engine.py`)
- **Hybrid retrieval.** Dense (`multilingual-e5-small`, `retriever.py`) and pure-Python Okapi
  **BM25** (`sparse.py`) fail differently — meaning vs exact tokens. They are fused by
  **Reciprocal Rank Fusion** (`hybrid.py`, `k_rrf=60`): combine *rankings*, not scores, so no
  scale normalisation is needed.
- **Two-stage reranking.** A bi-encoder approximates relevance; a **cross-encoder** reads
  query+passage together and judges directly, but is too slow corpus-wide. So hybrid retrieves a
  30-candidate pool and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reorders it, keeping the
  top *k* (`rerank.py`, `cross_encoder.py`). This is what breaks the recall ceiling (§5).
- **Grounded generation.** The generator answers **only** from the retrieved passages, cites
  them `[n]`, and emits an exact refusal string when the answer is absent — making refusals
  detectable downstream.

## 5. Evaluation strategy

The contract every change is held to (`src/boe_rag/eval/`):
- **Metrics** are pure-Python and dependency-free (`metrics.py`): recall@k, precision@k,
  hit-rate@k, MRR, nDCG@k. Recall/hit-rate say *whether* the answer was found; MRR/nDCG say
  *how well it was ranked* — the distinction matters once recall saturates.
- **Gold vs silver.** 20 hand-curated questions are the source of truth (the CI fixture). A
  generator (`generate.py`) prompts an LLM for self-contained questions grounded in sampled
  chunks, drops deictic/trivial ones, and keeps only answers an **LLM-judge** rates faithful —
  a 1,749-example *silver* set. Synthetic questions can flatter their generator, so the two
  tiers are reported separately. Both are published: `gonzalonao/boe-rag-evalset`.
- **End-to-end quality.** An LLM-as-judge (`judge.py`) scores each answer for **faithfulness**
  and **correctness** (0–1, `temperature=0`), plus refusal rate.
- **Measured results.** Dense → +hybrid → +rerank lifts recall@10 0.900 → **1.000** and MRR
  0.749 → **0.888**; article chunking beats fixed-size by **+0.063 MRR** while uniquely keeping
  exact citations. E2E baseline: faithfulness **0.990**, correctness **0.895**.
- **Uncertainty, quantified.** `eval/stats.py` reports a 95% bootstrap CI for recall@k and MRR
  on every run (wide on 20 gold questions — recall@10 0.900 [0.750, 1.000]; ~10× tighter on the
  1,749 silver examples) and a **paired sign-flip permutation test** for comparing two systems on
  the same queries — so a change is judged *significant*, not just numerically larger.
- **Adversarial security.** `eval/security.py` red-teams the generator against prompt
  injection, system-prompt exfiltration (canary-based detection), citation spoofing, and
  out-of-corpus hallucination — deterministic rule-based checks, no LLM judge. The baseline
  (9/14) deliberately surfaces a real weakness — fabricated citations — that motivates the
  Phase 5 citation-validation guardrail. The generator is hardened to treat passages and the
  question as data, not instructions.
- **CI regression gate.** `check_eval_regression.py` fails a PR if recall@10 or MRR drops more
  than a tolerance below the committed `eval_data/retrieval_baseline.json`.

## 6. Serving and operations (`service/`)
- **One engine, two surfaces.** FastAPI (`api.py`: `/ask`, `/search`, `/health`) and a Gradio
  chat UI (`ui.py`) are mounted on the **same app** over the **same** `Engine` instance, so the
  demo and the JSON API can never drift.
- **Resilience.** A bounded-LRU answer cache; a fixed-window per-IP rate limiter scoped to
  `/ask` + `/search`; graceful degradation — when the LLM tier is rate-limited, `/ask` returns
  the retrieved passages instead of failing, and `/search` never needed an LLM.
- **Provider layer (`llm/`).** Vendor-agnostic `LLMProvider` protocol; a `FallbackProvider`
  chains OpenRouter → Groq → Gemini with a **time-based circuit breaker** that skips a
  rate-limited provider for a cool-down and raises a *distinct* error when all are cooling down,
  so bulk callers can back off rather than die.
- **Cheap cold start.** The Docker image bakes in CPU-only PyTorch, the corpus, **precomputed E5
  embeddings**, and model weights, so the running Space does no downloads and no startup encode.
  Each version tag triggers `deploy-space.yml`, which mirrors to the Space — every deploy is a
  tagged release.

## 7. Observability (`service/tracing.py`)
Each stage is wrapped in a `Tracer` span (`answer → retrieve → rerank → generate`). Default is a
**no-op** (zero cost/deps). With `LANGFUSE_*` set (and the `obs` extra), a `LangfuseTracer`
exports the same spans to Langfuse; nested stages auto-parent into one trace named after the
question. The adapter is injected and fake-tested, so the dependency and network egress stay out
of CI. Quality *scores* (LLM-judge faithfulness, 👍/👎) are a planned follow-up.

## 8. Testing strategy
- The HTTP layer, UI helpers, engine, reranker, providers, and eval logic are all tested with
  **fakes** behind their protocols — the full suite runs with no models, no torch, no API keys.
- Pure-Python metrics + the regression gate run in CI as a quality gate. The heavy retrieval
  eval runs in a separate `eval-gate` CI job that fetches the published corpus and caches the
  embedding model.
- Strict quality bar: `ruff format` + `ruff check` + `mypy --strict` + `pytest`, enforced by
  pre-commit and CI.

## 9. Scaling roadmap and trade-offs

| Current choice | Why (now) | Planned swap (when) | Seam |
|---|---|---|---|
| In-memory NumPy dense + BM25 | 2,225 chunks fit in RAM; zero infra | **Qdrant** on-disk (dense+sparse) | `Searcher` |
| Off-the-shelf `multilingual-e5-small` | strong baseline, no training | **fine-tuned** Spanish-legal E5 + ONNX int8 | `Embedder` |
| sentence-transformers cross-encoder | accurate, simple | **ONNX int8** cross-encoder | `Reranker` |
| 2024 corpus slice | fast to iterate | **2015–present** Disposiciones Generales | corpus artifact |
| Custom metrics + LLM-judge | transparent, dependency-free | **RAGAS** alongside | `eval/` |

Each swap is a constructor change in `build_engine`, not a rewrite — the point of the protocol
seams. Corpus expansion is the keystone: it removes the metric saturation that currently caps
measurement and is the credibility prerequisite for the embedding fine-tune.

## 10. Key decisions log
- **Article-level chunking** over fixed-size windows — equal ranking quality, *plus* free exact
  citations. Measured, not assumed.
- **RRF over score fusion** — avoids normalising incomparable score scales.
- **Two-stage retrieve-then-rerank** — the only affordable way to use a cross-encoder.
- **Protocols + dependency injection** — the single decision that makes the system both testable
  in CI and cheap to evolve.
- **Eval set committed/published, gold as CI fixture** — quality is version-controlled like code.
