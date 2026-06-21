# Embedding fine-tune — measured, didn't ship (Arc 6)

**Question:** does contrastively fine-tuning `intfloat/multilingual-e5-small` on
mined BOE retrieval pairs beat the off-the-shelf model on Spanish-legal
retrieval? **Answer: no — and we can now say so with statistical confidence, so
we keep the off-the-shelf model.** This is the honest negative result; the value
is in the rigour of how it was reached, not in shipping a marginal win.

## Method

- **Training signal.** Mine `(question, positive-chunk, hard-negatives)` triples
  from the ~1.7k-example silver eval set (`scripts/finetune_embeddings.py`,
  `eval/mine_pairs.py`). Hard negatives are BM25 near-misses — the confusable
  passages a dense model most needs to separate.
- **Objective.** `MultipleNegativesRankingLoss` (in-batch + mined negatives) on
  `multilingual-e5-small`, 3 epochs, batch 64 (fit on a 12 GB RTX 5070 via
  `--max-seq-length 256 --grad-checkpointing`), fp16, `no_duplicates` sampler.
- **Decision gate.** `scripts/compare_models.py` scores base vs tuned on a
  held-out test set and judges the delta with a paired bootstrap CI + sign-flip
  permutation test (`eval/stats.py`). Ship only on a *significant* recall@10 gain.

## Getting the evaluation right mattered more than the model

The first runs evaluated on the 20-query **gold** set and returned Δrecall@10 =
`0.000` with CI `[0, 0]` — not because the model did nothing, but because that
set is too small and structurally pinned to detect a change:

- Two of its misses are **un-fixable by embeddings**: `q003`'s gold chunk is
  byte-identical to 20+ other `::0004` boilerplate sections (same vector, same
  `0.913143` score — a tie no re-embedding can break), and `q018` is genuinely
  hard. With only 20 queries and two stuck, recall@10 cannot move.

So the gate was **underpowered**, not the experiment conclusive. The fix was a
properly-sized, leakage-free test set: split the silver set **by source
document** (`eval/split.py`, union-find over co-occurring documents) so no test
positive comes from a document the model trained on, then retrain on the train
split and evaluate on ~350 held-out queries.

## Result (powered: 350 held-out queries, document-disjoint from training)

| Metric | Baseline | Candidate | Δ | 95% CI | p |
|---|---|---|---|---|---|
| Recall@10 | 0.931 | 0.929 | −0.003 | [−0.031, +0.023] | 1.000 |
| MRR | 0.767 | 0.780 | +0.013 | [−0.019, +0.043] | 0.427 |

For reference, the underpowered gold-20 runs: recall@10 flat (Δ=0 both times);
MRR −0.011 (crippled batch-16 run) → +0.023 (fair batch-64 run).

## Conclusion

- **No meaningful retrieval gain.** The recall@10 CI is a tight `[−3.1%, +2.3%]`
  around zero — at n=350 this *rules out* a real improvement rather than merely
  failing to find one. MRR leans slightly positive (+1.3pp, and did so in both
  fair runs), so the tune nudges ranking the right way, but the effect is small
  and not significant. Not worth a model-publishing pipeline.
- **Off-the-shelf `multilingual-e5-small` is already strong on legal Spanish.**
  That is the finding: the base model generalises well enough that domain tuning
  on this data doesn't pay off. We keep it; `E5Embedder`'s default is unchanged.
- **The gate did its job.** A significance-tested go/no-go, plus a leakage-aware
  test split sized for real power, prevented shipping a marginal/illusory win —
  which is the point of building the harness in the first place.

## Reproduce

```powershell
python scripts/split_evalset.py --in eval_data/generated_evalset.jsonl `
    --train-out eval_data/silver_train.jsonl --test-out eval_data/silver_test.jsonl `
    --test-fraction 0.2 --seed 42
python scripts/finetune_embeddings.py --corpus data/corpus/boe-2015-present.parquet `
    --train-evalset eval_data/silver_train.jsonl --out models/boe-e5-small `
    --epochs 3 --batch-size 64 --num-negatives 4 --max-seq-length 256 --grad-checkpointing
python scripts/compare_models.py --corpus data/corpus/boe-2015-present.parquet `
    --evalset eval_data/silver_test.jsonl --candidate-model models/boe-e5-small `
    --out reports/finetune_compare_silver
```

## If revisited

The lever now is *not* more training but a stronger signal: a larger, cleaner
labelled set (human-reviewed positives, deduplicated `::0004` boilerplate), or a
harder eval that the off-the-shelf model actually struggles with. LoRA / more
epochs on the same noisy silver pairs is unlikely to change the verdict.
