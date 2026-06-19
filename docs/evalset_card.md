---
language:
  - es
license: other
license_name: boe-reuse
license_link: https://www.boe.es/datosabiertos/documentos/Aviso_legal_reutilizacion.pdf
pretty_name: BOE RAG Assistant — Evaluation Set
size_categories:
  - 1K<n<10K
task_categories:
  - text-retrieval
  - question-answering
tags:
  - legal
  - spanish
  - rag
  - boe
  - evaluation
  - retrieval
configs:
  - config_name: seed
    data_files: "seed_evalset.jsonl"
    default: true
  - config_name: generated
    data_files: "generated_evalset.jsonl"
---

# BOE RAG Assistant — Evaluation Set

Question–answer pairs with chunk-level relevance judgments for evaluating retrieval
and grounded generation over the
[BOE corpus](https://huggingface.co/datasets/gonzalonao/boe-corpus). Every example
names the corpus chunk(s) that answer it, so it doubles as a retrieval relevance set
and as a grounded-QA reference.

Built for the [BOE RAG Assistant](https://github.com/gonzalonao/boe-rag-assistant)
project, where the **seed** split is the CI regression fixture.

## Splits

| Config | Examples | Provenance | Use |
|---|---|---|---|
| `seed` *(default)* | 20 | Hand-curated gold set | Source of truth; the CI eval-regression gate runs against it |
| `generated` | 1,749 | LLM-generated, faithfulness-filtered ("silver") | Scales statistical power for retrieval/generation experiments |

Keep the two apart: `seed` is the trustworthy ground truth you regression-test on;
`generated` is larger but noisier and is meant for higher-power A/B comparisons, not
as a release gate.

## Schema

Both splits share one JSON-Lines schema:

| Field | Type | Description |
|---|---|---|
| `example_id` | string | Stable id (`q001…` for seed, `gen-{chunk_id}` for generated). |
| `question` | string | A self-contained question in Spanish. |
| `relevant_chunk_ids` | list[string] | Corpus `chunk_id`s that answer the question (the relevance judgment). |
| `answer` | string | Reference answer grounded in the relevant chunk(s). |
| `category` | string | Topic label (seed) or `"generated"` (silver). |
| `difficulty` | string | Difficulty label (seed) or `"auto"` (silver). |

All `relevant_chunk_ids` resolve to a row in the
[`gonzalonao/boe-corpus`](https://huggingface.co/datasets/gonzalonao/boe-corpus)
2024 slice (verified: 0 dangling ids, 0 blank fields).

## How it was built

- **Seed (gold):** hand-written questions over real 2024 BOE provisions, each with a
  manually verified relevant chunk and reference answer, spanning topics
  (social security, energy, public finance, taxation, foreign affairs, culture,
  transport, justice, interior, electoral…).
- **Generated (silver):** `scripts/generate_evalset.py` samples substantive corpus
  chunks, prompts an LLM for a self-contained question + answer grounded in each
  chunk, discards deictic/low-quality questions, and **keeps only answers an
  LLM-judge rates faithful** to their source chunk. The chunk a question was derived
  from becomes its relevance judgment.

## Baseline numbers

Dense retrieval with `intfloat/multilingual-e5-small` over the 2024 corpus
(`scripts/run_eval.py`, k = 10):

| Split | Queries | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| `seed` | 20 | 0.90 | 0.749 | — |
| `generated` | 1,749 | 0.963 | 0.827 | 0.860 |

The silver split scores higher because each question is derived from a single chunk,
which makes the target easier to recover; treat it as a high-power relative-comparison
instrument, and the seed split as the absolute bar.

## Limitations

- Relevance is single-source: a question's judgment is the chunk it came from
  (silver) or the curator's pick (seed); other chunks could also be relevant but are
  not labelled, so recall is a lower bound.
- The silver split inherits any LLM biases not caught by the faithfulness filter.
- Coverage tracks the underlying corpus, currently the **2024** BOE slice.

## Licensing

Derived from BOE content, publicly reusable under Spanish
reuse-of-public-sector-information rules (Law 37/2007 and RD 1495/2011); see the
linked legal notice. Always consult [boe.es](https://www.boe.es) for authoritative
text. Questions/answers are research artifacts, not legal advice.

## Citation

```
@misc{boe_rag_evalset,
  title  = {BOE RAG Assistant — Evaluation Set},
  author = {López Crespo, Gonzalo},
  year   = {2026},
  url    = {https://huggingface.co/datasets/gonzalonao/boe-rag-evalset}
}
```
