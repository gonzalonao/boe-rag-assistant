# The Project in Plain Words

This document explains, without jargon, what the BOE RAG Assistant is, the problem
it solves, and what each part of it does. If you read only one file to understand
the project, read this one.

---

## 1. The one-sentence version

We are building a question-answering assistant for **Spanish law**: you ask a
question in normal language ("how many days do I have to appeal an administrative
decision?") and it answers you, quoting the exact article of the exact law its
answer comes from, with a link to the official source.

## 2. The problem it solves

Spanish legislation is published every day in the **BOE** (*Boletín Oficial del
Estado* — the official state gazette). It is:

- **Huge** — decades of laws, decrees, and regulations.
- **Hard to read** — dense legal language, cross-references everywhere.
- **Hard to search** — keyword search returns documents, not answers.

A normal person (or even a busy professional) can't easily find "the rule that
applies to my situation." Our assistant reads the law for you and gives a direct,
**sourced** answer. The "sourced" part is the whole point: an answer you can't
verify is worthless in a legal context, so every answer points to the article it
came from.

## 3. What is "RAG"? (the core idea)

RAG stands for **Retrieval-Augmented Generation**. It's a recipe for making an AI
answer questions about a specific body of knowledge *without lying*.

The naive alternative would be to just ask a chatbot "what does Spanish law say
about X?" — but a chatbot makes things up (it "hallucinates"), and it doesn't know
the latest laws. RAG fixes both problems with a simple trick, in three steps:

1. **Retrieve** — first, *search* our collection of laws and pull out the handful
   of passages most relevant to the question. (Like a librarian fetching the right
   pages before anyone tries to answer.)
2. **Augment** — hand those passages to the AI as context: "Here are the relevant
   legal texts. Use ONLY these to answer."
3. **Generate** — the AI writes the answer grounded in those passages, and cites
   them.

So the AI never answers from memory — it answers from documents we retrieved and
showed it. That's what makes the answer trustworthy and current.

> Analogy: it's an open-book exam instead of a memory test. We make sure the AI
> has the right book open to the right page before it writes anything.

## 4. The big picture (how data flows)

```
  STEP A: build the knowledge base (done once, refreshed over time)
  BOE website ──▶ download laws ──▶ clean & split into pieces ──▶ store as a searchable index

  STEP B: answer a question (every time a user asks)
  question ──▶ search the index ──▶ pick the best passages ──▶ AI writes a cited answer
```

Everything in the project is one of these two flows: **building the knowledge
base** (offline) or **answering questions** (online).

## 5. The components, one by one

### Already built — the **ingestion pipeline** (building the knowledge base)

This is the machinery that turns the raw BOE website into a clean, searchable
collection. It lives in `src/boe_rag/ingest/` and has four parts:

- **The client** (`client.py`) — the part that *downloads* from the BOE's official
  data service. It's polite (doesn't hammer their servers) and stubborn in a good
  way (if a download fails because the server is briefly busy, it waits and retries
  instead of giving up).

- **The parser** (`parser.py`) — the BOE gives us messy, deeply-nested data files.
  The parser *reads and tidies* them: it pulls out the useful facts (title, date,
  type of law, the actual text) and throws away the noise.

- **The chunker** (`chunker.py`) — this is the clever bit. We can't feed a whole
  300-page law to the search system; we have to cut it into **chunks**. A dumb
  approach cuts every 500 words regardless of meaning. Ours cuts **along the law's
  own structure** — one chunk per *article*, while remembering which Title and
  Chapter that article sits under. That's why the assistant can say "according to
  *Ley 39/2015, Artículo 3*" instead of "according to some document, somewhere."

- **The corpus writer** (`corpus.py`) — saves all those chunks into a single
  efficient file format (**Parquet**) ready to be published and searched.

Tying them together: the **pipeline** (`pipeline.py`) and a command-line tool
(`boe-ingest`) that says "ingest everything from this date to that date."

### Coming next — the parts not built yet

- **The eval harness** (Phase 2, *next*) — before we make the search "smart," we
  build a **scorecard**. We write a set of real questions with known correct
  answers, then measure how often the system finds the right passage. This is the
  professional discipline at the heart of the project: **every future improvement
  has to prove itself by moving a number on this scorecard**, or we don't keep it.
  Most hobby projects skip this; serious ones don't.

- **The search index + retrieval** (Phase 3) — the actual "search engine" over the
  chunks. It combines two ways of searching: **keyword matching** (good for exact
  terms like a law number) and **meaning matching** (good for "appeal a decision"
  finding text about "recurso administrativo" even though the words differ).
  *Meaning matching* works by turning text into lists of numbers called
  **embeddings**, where similar meanings end up with similar numbers.

- **Reranking** (Phase 3) — after the search returns ~30 candidate passages, a
  second, more careful model re-sorts them so the very best ones are on top.

- **Fine-tuning** (Phase 4) — we take a general-purpose "meaning matching" model
  and **train it specifically on Spanish legal text** so it gets noticeably better
  at this domain. This is the step that runs on your home GPU.

- **Generation** (Phase 5) — the part that actually writes the answer, with strict
  rules: cite your sources, and if the documents don't contain the answer, say "I
  don't know" instead of inventing one.

- **The web app** (Phase 6) — a simple website where anyone can type a question and
  see the cited answer. It runs free on **Hugging Face Spaces**.

- **Auto-updates** (Phase 7) — a scheduled job that adds each new day's BOE to the
  knowledge base automatically, so the assistant never goes stale.

## 6. Where the public artifacts live

Two homes, on purpose:

- **GitHub** (`github.com/gonzalonao/boe-rag-assistant`) — the *code*.
- **Hugging Face Hub** — the *data and AI models*. This is the standard place to
  publish datasets and models in the AI world. We'll publish: the law collection
  (dataset), the question scorecard (dataset), the fine-tuned model, and the live
  demo (Space).

## 7. Why this is a strong portfolio piece

It shows the whole lifecycle of a real AI product, not just a notebook:
data engineering (ingestion), measurement-driven development (the scorecard),
model training (fine-tuning), and shipping a live, working app — all on a real,
useful, Spanish-language problem, built for $0.

## 8. Current status

- ✅ **Phase 0** — project setup, automated quality checks.
- ✅ **Phase 1** — ingestion pipeline (this document's "already built" section).
- ⏭️ **Phase 2** — the eval scorecard (next).

See `README.md` for the technical version and `RAG-ASSISTANT-PROJECT-PLAN.md`
(in the workspace) for the full roadmap.
