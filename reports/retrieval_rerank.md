# Retrieval evaluation — reranking

- **Generated:** 2026-06-26 17:38 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Corpus:** `boe-2015-present.parquet` (25419 chunks)
- **Queries:** 20
- **Retrieved per query:** 20
- **Rerank pool:** 30

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| dense | 0.900 | 0.090 | 0.900 | 0.691 | 0.740 |
| hybrid (RRF) | 0.850 | 0.085 | 0.850 | 0.746 | 0.766 |
| hybrid + cross-encoder | 1.000 | 0.100 | 1.000 | 0.862 | 0.894 |
