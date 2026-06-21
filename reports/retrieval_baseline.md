# Retrieval evaluation — baseline

- **Generated:** 2026-06-21 00:29 UTC
- **Model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2015-present.parquet` (25419 chunks)
- **Queries:** 20
- **Retrieved per query:** 20

## Metrics @10

Confidence intervals are 95% bootstrap (per-query resampling).

| Metric | Value | 95% CI |
|---|---|---|
| Recall@10 | 0.850 | [0.700, 1.000] |
| Precision@10 | 0.085 | — |
| Hit rate@10 | 0.850 | — |
| MRR | 0.674 | [0.495, 0.835] |
| nDCG@10 | 0.715 | — |

## Per-question first-hit rank

| Example | First relevant rank |
|---|---|
| q001 | 13 |
| q002 | 2 |
| q003 | MISS |
| q004 | 1 |
| q005 | 2 |
| q006 | 1 |
| q007 | 1 |
| q008 | 5 |
| q009 | 1 |
| q010 | 2 |
| q010b | 1 |
| q011 | 1 |
| q012 | 1 |
| q013 | 1 |
| q014 | 5 |
| q015 | 1 |
| q016 | 2 |
| q017 | 1 |
| q018 | MISS |
| q019 | 1 |

**Misses (2):** q003, q018
