---
language:
  - es
license: other
license_name: boe-reuse
license_link: https://www.boe.es/datosabiertos/documentos/Aviso_legal_reutilizacion.pdf
pretty_name: BOE Disposiciones Generales — RAG Corpus
size_categories:
  - 10K<n<100K
task_categories:
  - text-retrieval
  - question-answering
tags:
  - legal
  - spanish
  - rag
  - boe
configs:
  - config_name: default
    data_files: "data/*.parquet"
---

# BOE Disposiciones Generales — RAG Corpus

A chunked, retrieval-ready corpus of the **"I. Disposiciones Generales"** section
of Spain's *Boletín Oficial del Estado* (BOE). Each row is one structure-aware
chunk — typically a single article — carrying the legal hierarchy it belongs to,
so retrieved passages can be cited precisely (e.g. *"Ley 39/2015, Artículo 3"*).

Built for the [BOE RAG Assistant](https://github.com/gonzalonao/boe-rag-assistant)
project.

## Why this corpus

Most open RAG corpora are English. This one targets a real, high-value, daily-updated
Spanish legal source and preserves its structure rather than slicing it into
fixed-size windows, which makes retrieval both more accurate and verifiable.

## Schema

| Column | Type | Description |
|---|---|---|
| `chunk_id` | string | Deterministic id `{document_id}::{ordinal}`. |
| `document_id` | string | BOE document identifier (e.g. `BOE-A-2024-714`). |
| `document_title` | string | Official document title. |
| `text` | string | The chunk text (article body and heading). |
| `ordinal` | int | Position of the chunk within its document. |
| `titulo` | string? | Enclosing TÍTULO heading, when present. |
| `capitulo` | string? | Enclosing CAPÍTULO heading, when present. |
| `seccion` | string? | Enclosing SECCIÓN heading, when present. |
| `articulo` | string? | Enclosing article heading, when present. |
| `citation` | string | Human-readable citation for the span. |
| `url_html` | string | Link to the official document on boe.es. |

## Collection methodology

- **Source:** the official [BOE Open Data API](https://www.boe.es/datosabiertos/)
  (daily *sumario* index → per-document XML).
- **Scope:** section `1` ("I. Disposiciones Generales"), the most durable and
  citable content.
- **Processing:** XML parsed into structured records, then chunked along the
  legal hierarchy (título → capítulo → sección → artículo). Whitespace
  normalised; blank and sub-minimum fragments merged into neighbours.
- **Reproducible:** regenerate with the project's `boe-ingest` CLI.

## Licensing

BOE content is publicly reusable under Spanish reuse-of-public-sector-information
rules (Law 37/2007 and RD 1495/2011); see the linked legal notice. Attribution to
the BOE is required and the data must not be altered in a way that misrepresents
its official meaning. This dataset is a derived, reformatted copy for research and
retrieval; always consult [boe.es](https://www.boe.es) for the authoritative text.

## Citation

```
@misc{boe_rag_corpus,
  title  = {BOE Disposiciones Generales — RAG Corpus},
  author = {López Crespo, Gonzalo},
  year   = {2026},
  url    = {https://huggingface.co/datasets/gonzalonao/boe-corpus}
}
```
