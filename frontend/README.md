# BOE RAG — Frontend

A small React + Vite (TypeScript) single-page app for the BOE RAG Assistant. It calls the
FastAPI JSON API (`POST /ask`) and renders the grounded answer with its cited BOE sources.

Deployed **separately** from the API (the Space serves the JSON API; this SPA is hosted on a
static host and talks to it cross-origin — the API allows the frontend origin via
`BOE_CORS_ORIGINS`).

## Stack

- **React 18 + Vite 5**, **TypeScript** (`strict`, plus `noUncheckedIndexedAccess` and
  `exactOptionalPropertyTypes`).
- **Biome** for one-tool lint + format (mirrors the Python side's "Ruff only").
- No runtime UI dependencies beyond React — plain CSS, system fonts.

## Develop

```bash
cd frontend
npm install
cp .env.example .env.local          # set VITE_API_BASE_URL to the API origin
npm run dev                         # http://localhost:5173
```

Point `VITE_API_BASE_URL` at the deployed Space (default in `.env.example`) or at a local
`uvicorn boe_rag.service.app:app` started with `BOE_CORS_ORIGINS=http://localhost:5173`.

## Quality

```bash
npm run lint        # Biome check (lint + format verify)
npm run typecheck   # tsc --noEmit, strict
npm run build       # typecheck + production build to dist/
```

## Deploy (GitHub Pages)

`.github/workflows/deploy-frontend.yml` builds this app and publishes it to GitHub Pages when
frontend changes land on `main` (or via *Run workflow*). It sets `VITE_BASE=/boe-rag-assistant/`
(the project-site subpath) and `VITE_API_BASE_URL` to the HF Space (override with the
`API_BASE_URL` repo variable).

One-time setup:

1. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Add the published origin `https://gonzalonao.github.io` to the API's `BOE_CORS_ORIGINS`
   (HF Space secret/variable) so the browser is allowed to call it cross-origin, then redeploy
   the Space.

`npm run build` emits a static `dist/`; the same artifact deploys to any other static host
(Vercel/Netlify) — set `VITE_API_BASE_URL` there and serve from the root (`VITE_BASE` unset).

## Layout

```
src/
  api.ts                 typed fetch client for /ask (user-facing error messages)
  types.ts               mirrors the FastAPI Pydantic models
  App.tsx                state machine: idle → loading → done/error
  components/
    AskForm.tsx          question input + example prompts
    AnswerPanel.tsx      grounded answer or refusal notice
    SourceCard.tsx       one cited passage, linked to boe.es
  styles.css             theme + layout
```
