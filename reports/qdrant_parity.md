# Qdrant backend parity — exact vs the in-memory NumPy index

**Question:** does serving the dense leg from Qdrant (`QdrantSearcher`) instead of
the in-memory NumPy index (`DenseRetriever`) change retrieval quality? It should
not — both rank the *same* precomputed E5 vectors by cosine, so a swap must be
faithful.

## Setup

- Same artifact for both: `data/corpus/boe-2015-present-embeddings.npz`
  (25,419 × 384, L2-normalised — every row norm `1.00000`).
- Same 20-query gold set (`eval_data/seed_evalset.jsonl`), `k=10`, `retrieve_n=20`.
- Qdrant run in local embedded mode with **exact** search
  (`run_eval.py --qdrant-path … --qdrant-exact`).

## Result

| Backend | Recall@10 | MRR | Misses |
|---|---|---|---|
| NumPy (`DenseRetriever`) | 0.850 | 0.674 | q003, q018 |
| Qdrant (exact) | 0.900 | 0.684 | q018 |

19 of 20 queries rank **identically**. The only difference is **q003**, and it is
not a quality difference — it is a tie-break.

## Root cause: a tie among duplicate embeddings (not a backend gap)

q003's labelled answer `BOE-A-2024-714::0004` shares an **identical embedding**
with 20+ other chunks. They are all the `::0004` section of different BOE
documents — textually identical boilerplate — so they collapse to the exact same
vector and the **same cosine score, `0.913143`**:

```
exact-cosine ranks 3..24 — every score == 0.913143
score(rank 5) − score(rank 22) = 0.000000
```

Both backends compute these scores identically (top-of-list scores `0.915202`,
`0.913198`, `0.913143…` match to 6 decimals across NumPy and Qdrant). Within the
flat tie band the order is arbitrary: NumPy's `argsort` (index order) places the
gold copy at rank 22; Qdrant places it at rank 5. Recall@10 therefore counts q003
as a miss for one and a hit for the other — pure tie-breaking on duplicated
content, with the **same underlying retrieval**.

## Conclusion

- **Backend parity holds exactly** — Qdrant and the NumPy index assign identical
  cosine scores to identical vectors. The swap (behind the `Searcher` seam) is
  faithful; `0.850` and `0.900` bracket the same result, with the gap living
  entirely inside a duplicate-embedding tie. The committed gate stays at the
  NumPy value (`0.85`), which is what the live Space serves.
- **Approximate vs exact:** Qdrant serves from its HNSW index by default
  (production-realistic, fast); `--qdrant-exact` / `connect_searcher(exact=True)`
  forces brute force for this parity check. Embedded local mode also warns it is
  not recommended above 20k points — a server/Cloud deployment is the production
  target.
- **Data-quality note (follow-up):** the corpus carries substantial duplicate
  boilerplate (many `::0004` sections embed identically). This inflates the index
  and makes gold labels on those sections ambiguous — a candidate for a
  dedup/canonicalisation pass and a look at the chunking of those sections.
