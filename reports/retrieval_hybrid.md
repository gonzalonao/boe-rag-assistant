# Retrieval evaluation — dense vs BM25 vs hybrid

- **Generated:** 2026-06-14 14:21 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2024.parquet` (2225 chunks)
- **Queries:** 20
- **Retrieved per query:** 20
- **Fusion:** Reciprocal Rank Fusion (k_rrf=60)

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| dense | 0.900 | 0.090 | 0.900 | 0.749 | 0.783 |
| bm25 | 0.900 | 0.090 | 0.900 | 0.732 | 0.773 |
| hybrid (RRF) | 0.900 | 0.090 | 0.900 | 0.763 | 0.793 |
