# Scaling runbook — corpus expansion → Qdrant → fine-tuned embeddings

The keystone arc of the roadmap. **Arc 4 (corpus expansion)** is actionable now and
documented step-by-step below. **Arc 5 (Qdrant)** and **Arc 6 (embedding fine-tune)**
are staged: their plans are concrete, but they are best built *after* the larger corpus
exists, since both are justified and measured against it.

All commands are Windows PowerShell, run from the repo root with the virtualenv's
interpreter (`.\.venv\Scripts\python.exe`). The App-Control note applies: invoke tools as
`python.exe -m <tool>`, never the bare `.exe` shim.

---

## Arc 4 — Corpus expansion (2015 → present)

**Why:** the corpus is currently the 2024 slice only, which saturates the gold metrics
and narrows coverage (e.g. missing foundational laws like Ley 39/2015). Widening to
2015–present removes that ceiling and is the credibility prerequisite for the fine-tune.

**One-time prep**

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev,ml,hub]
# Authenticate to the Hub. App-Control blocks the huggingface-cli .exe shim, so
# invoke it through the interpreter (it persists the token to disk afterwards):
.\.venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli login
```

> A `protobuf` dependency-conflict warning from pip (opentelemetry-proto vs the `ml`
> extra) is expected and harmless here — it only affects the optional `obs`/Langfuse
> path, which the crawl, publish, and embedding steps never touch.

**Step 1 — Crawl, year by year (resumable).**
This is the long step: the BOE API is crawled politely (~0.5 s/request), so a decade of
daily issues takes *hours*. It is resumable — each year is written to its own shard and a
re-run skips years already done, so you can stop with `Ctrl+C` and resume anytime.

```powershell
.\.venv\Scripts\python.exe scripts/ingest_corpus_years.py `
    --start-year 2015 --end-year 2026 `
    --out-dir data/corpus/years `
    --merged-out data/corpus/boe-2015-present.parquet
```

- Watch the per-year logs (`Year 2015 done: N chunks`).
- If it stops, just run the **same command again** — completed years are skipped.
- To only re-merge after all shards exist (no network), add `--merge-only`.

**Step 2 — Sanity-check the merged corpus.**

```powershell
.\.venv\Scripts\python.exe scripts/inspect_corpus.py data/corpus/boe-2015-present.parquet
```

Confirm the chunk count jumped (expect 10–50× the 2,225 of the 2024 slice) and spot-check a
few citations.

**Step 3 — Publish to the Hub** (same dataset repo; the new, larger Parquet becomes canonical).

```powershell
.\.venv\Scripts\python.exe scripts/push_corpus_to_hub.py `
    --parquet data/corpus/boe-2015-present.parquet `
    --repo-id gonzalonao/boe-corpus
```

> Update `docs/dataset_card.md`'s coverage/row-count lines first if they name the 2024 span.

**Step 4 — Precompute embeddings** for the new corpus (CPU works; the RTX 5070 is far faster
— set `CUDA` torch if you want GPU here).

```powershell
.\.venv\Scripts\python.exe scripts/precompute_embeddings.py `
    --corpus data/corpus/boe-2015-present.parquet `
    --out data/corpus/boe-2015-present-embeddings.npz
```

**Step 5 — Regenerate the CI regression baseline.** The gate in `eval_data/retrieval_baseline.json`
is pinned to the 2024 numbers; an intended corpus change is exactly when to move it.

```powershell
.\.venv\Scripts\python.exe scripts/run_eval.py `
    --corpus data/corpus/boe-2015-present.parquet `
    --evalset eval_data/seed_evalset.jsonl `
    --out reports/retrieval_baseline
```

Then edit `eval_data/retrieval_baseline.json`: update the `guards.recall_at_k.baseline` and
`guards.mrr.baseline` to the new `reports/retrieval_baseline.json` point values, and the
`corpus` field to name the new span. Commit on a `feat/corpus-2015-present` branch.

**Step 6 — Deploy to the Space.** The Docker build fetches the corpus from
`gonzalonao/boe-corpus` and precomputes embeddings at build time, so once Step 3 has
published the larger Parquet, a new release tag rebuilds the Space with it:

```powershell
git checkout main; git pull origin main      # after the develop→main PR is merged
git tag -a v0.3.0 -m "v0.3.0: corpus expansion 2015–present"
git push origin v0.3.0
gh run watch                                  # deploy-space.yml
```

> Build-time note: precompute on the build runner grows with the corpus. If the HF Space
> build times out on a much larger corpus, switch the image to **download** a prebuilt
> embeddings `.npz` (publish it alongside the corpus and `COPY`/fetch it) instead of
> computing it in `RUN`. Flag this to me and I'll adjust the Dockerfile.

**Definition of done:** larger corpus on the Hub, CI baseline updated and green, Space
serving the wider corpus, and the new recall/MRR recorded in `reports/`.

---

## Arc 5 — Qdrant swap (staged — build after Arc 4)

**Why now-ish:** once the corpus is 10–50× larger, the in-memory NumPy index stops being
the obvious choice; an on-disk vector DB is the honest production answer and puts a real
vector store on the CV. It pairs naturally with the bigger corpus.

**Plan (what I'll build):**
1. New `QdrantSearcher` implementing the existing `Searcher` protocol
   (`src/boe_rag/eval/retriever.py`) — same `search(query, k) -> [(chunk_id, score)]`
   contract, so `build_engine` swaps it in with a one-line constructor change.
2. An indexing script (`scripts/build_qdrant_index.py`) that upserts the precomputed E5
   vectors + chunk-id payloads into a local Qdrant (Docker) collection.
3. Config via `boe_rag.settings` (`QDRANT_URL`, `QDRANT_COLLECTION`); `qdrant-client` behind
   a new `qdrant` extra so CI stays lean.
4. Keep BM25 in memory and fuse with RRF exactly as today — only the dense leg moves.
5. Prove parity: re-run `run_eval.py` with the Qdrant dense leg and show metrics match the
   NumPy baseline (this is a *swap*, not a quality change), plus a latency note.

**Your side when it's ready:** run a local Qdrant (`docker run -p 6333:6333 qdrant/qdrant`),
run the index-build script once, set the two env vars in `.env`. I'll hand you exact steps.

---

## Arc 6 — Embedding fine-tune on the RTX 5070 (tooling BUILT — run on the GPU)

**Why:** the single most differentiating ML artifact — a domain-tuned Spanish-legal
embedding model with a measured before/after beats any off-the-shelf demo. The wider corpus
(Arc 4) gives the training signal; the held-out gold set keeps the win honest.

**What's built (on `develop`):**
- `src/boe_rag/eval/mine_pairs.py` — mines `(question, positive-chunk, hard-negatives)` pairs
  from the **silver** eval set (`eval_data/generated_evalset.jsonl`), with hard negatives
  drawn from BM25 near-misses (the confusable passages the dense model most needs to separate).
  Pure + unit-tested; also lays the pairs out as a rectangular, E5-prefixed training dataset.
- `scripts/finetune_embeddings.py` — `SentenceTransformerTrainer` +
  `MultipleNegativesRankingLoss` on `multilingual-e5-small`, 12 GB config (fp16,
  `no_duplicates` batch sampler). Needs the `ml` + `train` extras.
- `src/boe_rag/eval/compare.py` + `scripts/compare_models.py` — the go/no-go gate: scores
  base vs tuned on the **gold** set (`eval_data/seed_evalset.jsonl`, held out from training)
  and judges the difference with a paired bootstrap CI + sign-flip permutation test
  (`eval/stats.py`). Ships only on a significant recall@10 win (exit 0 = ship, 2 = no-ship).

> **Train/eval split:** train on the 1,749-example silver set, evaluate on the 20 hand-curated
> gold questions. The gold *queries* are held out; some corpus chunks may appear as positives
> in both (you are teaching the model the corpus), so the gold metric measures generalisation
> to unseen queries — noted honestly in the model card.

**Your side — the GPU run (detailed steps).** PowerShell from the repo root, venv interpreter.

1. Pull and install the training extras (on top of your existing cu128 torch):
   ```powershell
   git checkout develop; git pull origin develop
   .\.venv\Scripts\python.exe -m pip install -e ".[ml,train]"
   ```
2. Fine-tune (≈ a few minutes on the 5070; e5-small is 118 M params):
   ```powershell
   .\.venv\Scripts\python.exe scripts/finetune_embeddings.py `
       --corpus data/corpus/boe-2015-present.parquet `
       --train-evalset eval_data/generated_evalset.jsonl `
       --out models/boe-e5-small `
       --epochs 1 --batch-size 64 --num-negatives 4 `
       --pairs-out data/train/boe_pairs.jsonl
   ```
   Watch VRAM; if it OOMs, drop `--batch-size` to 32. Output model lands in `models/boe-e5-small`.
3. **Go/no-go** — score tuned vs base on the gold set (encodes the 25K corpus twice):
   ```powershell
   .\.venv\Scripts\python.exe scripts/compare_models.py `
       --corpus data/corpus/boe-2015-present.parquet `
       --evalset eval_data/seed_evalset.jsonl `
       --candidate-model models/boe-e5-small `
       --out reports/finetune_compare
   ```
   Read the printed verdict + `reports/finetune_compare.md`. **Send me the table.**
4. **Only if it ships** (significant recall@10 gain): I then wire publishing —
   `gonzalonao/boe-embeddings-e5` with the before/after model card, repoint `E5Embedder`'s
   default model, re-precompute the `.npz`, republish, and tag a release to redeploy. If it
   does **not** win, we keep the off-the-shelf model and record the honest null result (still
   a portfolio-worthy "measured, didn't ship" story) — then iterate (more epochs, more
   negatives, larger batch, or LoRA).

**Optional follow-on:** ONNX int8 export for CPU latency (Arc 7).

---

## Sequencing summary

```
Arc 4 (you crawl + I wire baseline/docs)  ─►  Arc 5 (I build, you run local Qdrant)
                                          └►  Arc 6 (I build, you run the GPU train)
```

Arc 4 unblocks both 5 and 6. Do 4 first; 5 and 6 can then proceed in parallel.
