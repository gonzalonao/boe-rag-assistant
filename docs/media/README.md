# Media assets

Drop two files here, then uncomment the matching image lines in the top-level `README.md`.
Both are referenced (commented out) so the README never shows a broken image before they exist.

## 1. `demo.gif` — the chat UI answering a question

Referenced from `README.md` → **Demo** section.

**How to record it (Windows 11):**

1. Start the service locally with an LLM key set:
   ```powershell
   $env:OPENROUTER_API_KEY = "sk-or-..."
   .\.venv\Scripts\python.exe -m uvicorn boe_rag.service.app:app
   ```
   Wait for the log line `Application startup complete`, then open `http://localhost:8000/`.
2. Press **Win + G** (Xbox Game Bar) → **Capture** widget → **Record**. (Game Bar records the
   focused window; alternatively use [ScreenToGif](https://www.screentogif.com/), which exports
   GIF directly and is easier to trim.)
3. Ask one clear question, e.g. *"¿Cuánto tiempo tiene la administración para resolver un
   procedimiento?"*, and let the cited answer render fully (show the linked sources).
4. Keep it **8–15 s**. Trim dead time at both ends.
5. Export as GIF, target **≤ 5 MB** and ~1000 px wide (ScreenToGif: reduce FPS to ~10 and
   colour count if it's too big). Save as `docs/media/demo.gif`.
6. In `README.md`, uncomment the `![BOE RAG Assistant demo](docs/media/demo.gif)` line.

## 2. `langfuse-trace.png` — a single request trace

Referenced from `README.md` → **Observability** section.

**How to capture it:**

1. Run the service with Langfuse enabled (see the README Observability section for the env
   vars and the `obs` extra), then ask one question through the UI or `/ask`.
2. Open your Langfuse project → **Tracing** → click the trace named after your question.
3. Expand the tree so all four spans are visible: `answer → retrieve → rerank → generate`.
4. Screenshot just that panel (Win + Shift + S → window/region). Save as
   `docs/media/langfuse-trace.png` (PNG, ≤ ~400 KB; crop to the trace, no browser chrome).
5. In `README.md`, uncomment the `![Langfuse trace ...](docs/media/langfuse-trace.png)` line.

> Keep both files small — they live in git. If either grows past ~1 MB, optimise it
> (fewer colours/FPS for the GIF, PNG compression for the screenshot) rather than committing
> a heavy asset.
