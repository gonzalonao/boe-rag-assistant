# TypeScript / Frontend — Code Standards

Applies to the `frontend/` React + Vite SPA.

## Formatting & linting

- **Biome only** — one tool for both format and lint (mirrors the Python side's "Ruff only").
  `npm run format` writes, `npm run lint` verifies. Configure in `biome.json`.
- 2-space indent, double quotes, trailing commas, line width 88.

## Type checking

- **`strict` is mandatory**, plus `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`.
  Code must pass `tsc --noEmit` (`npm run typecheck`) with zero errors.
- No `any`. Prefer `unknown` + narrowing. API response shapes live in `src/types.ts` and must
  mirror the FastAPI Pydantic models (`src/boe_rag/service/models.py`) — the API is the source
  of truth.

## Components & structure

- Function components only; hooks for state. One component per file, named export.
- Keep network/IO in `src/api.ts`; components stay presentational and take typed props.
- No runtime UI framework dependencies beyond React unless justified — plain CSS first.
- User-facing copy is Spanish (the assistant's domain); code, comments, and identifiers English.

## Errors

- Never swallow a failed request silently. `api.ts` maps non-2xx/network failures to an
  `ApiError` carrying a user-facing message; components render it.

## Dependencies

- Pin exact versions in `package.json` (no `^`/`~`), matching the Python side's pinning rule.
