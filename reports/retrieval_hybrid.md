# Retrieval evaluation — dense vs BM25 vs hybrid

- **Generated:** 2026-06-26 17:20 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2015-present.parquet` (25419 chunks)
- **Queries:** 20
- **Retrieved per query:** 20
- **Fusion:** Reciprocal Rank Fusion (k_rrf=60)

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| dense | 0.900 | 0.090 | 0.900 | 0.691 | 0.740 |
| bm25 | 0.900 | 0.090 | 0.900 | 0.678 | 0.732 |
| hybrid (RRF) | 0.850 | 0.085 | 0.850 | 0.746 | 0.766 |
