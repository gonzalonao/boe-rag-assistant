# Retrieval evaluation - chunking strategies

- **Generated:** 2026-06-15 01:11 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2024.parquet` (105 documents)
- **Queries:** 20
- **Relevance:** document-level (hit = chunk from a relevant document)
- **Retrieved per query:** 50
- **Fixed window:** 1000 chars, 150 overlap

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| article (current) | 1.000 | 0.100 | 1.000 | 0.975 | 0.982 |
| fixed-size | 1.000 | 0.100 | 1.000 | 0.912 | 0.935 |
| whole-document | 1.000 | 0.100 | 1.000 | 0.975 | 0.982 |
