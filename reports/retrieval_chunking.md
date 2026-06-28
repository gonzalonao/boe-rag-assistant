# Retrieval evaluation - chunking strategies

- **Generated:** 2026-06-26 17:40 UTC
- **Embedding model:** `intfloat/multilingual-e5-small`
- **Corpus:** `boe-2015-present.parquet` (1043 documents)
- **Queries:** 20
- **Relevance:** document-level (hit = chunk from a relevant document)
- **Retrieved per query:** 50
- **Fixed window:** 1000 chars, 150 overlap

## Metrics @10

| Retriever | Recall | Precision | Hit rate | MRR | nDCG |
|---|---|---|---|---|---|
| article (current) | 0.950 | 0.095 | 0.950 | 0.902 | 0.913 |
| fixed-size | 0.950 | 0.095 | 0.950 | 0.877 | 0.895 |
| whole-document | 0.950 | 0.095 | 0.950 | 0.824 | 0.849 |
