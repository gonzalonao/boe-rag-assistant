# Retrieval evaluation — baseline

- **Generated:** 2026-06-20 01:20 UTC
- **Model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2024.parquet` (2225 chunks)
- **Queries:** 20
- **Retrieved per query:** 20

## Metrics @10

Confidence intervals are 95% bootstrap (per-query resampling).

| Metric | Value | 95% CI |
|---|---|---|
| Recall@10 | 0.900 | [0.750, 1.000] |
| Precision@10 | 0.090 | — |
| Hit rate@10 | 0.900 | — |
| MRR | 0.749 | [0.584, 0.900] |
| nDCG@10 | 0.783 | — |

## Per-question first-hit rank

| Example | First relevant rank |
|---|---|
| q001 | 12 |
| q002 | 2 |
| q003 | 1 |
| q004 | 1 |
| q005 | 1 |
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

**Misses (1):** q018
