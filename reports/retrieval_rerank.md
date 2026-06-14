# Retrieval evaluation — reranking

- **Generated:** 2026-06-14 16:06 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Corpus:** `boe-2024.parquet` (2225 chunks)
- **Queries:** 20
- **Retrieved per query:** 20
- **Rerank pool:** 30

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| dense | 0.900 | 0.090 | 0.900 | 0.749 | 0.783 |
| hybrid (RRF) | 0.900 | 0.090 | 0.900 | 0.763 | 0.793 |
| hybrid + cross-encoder | 1.000 | 0.100 | 1.000 | 0.888 | 0.913 |
