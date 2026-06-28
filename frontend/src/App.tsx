import { useState } from "react";
import { ApiError, ask } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { AskForm } from "./components/AskForm";
import type { AnswerResponse } from "./types";

type Status = "idle" | "loading" | "done" | "error";

export function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState<string | null>(null);

  async function handleAsk(question: string) {
    setStatus("loading");
    setError(null);
    setResult(null);
    setAsked(question);
    try {
      const response = await ask(question);
      setResult(response);
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Ha ocurrido un error inesperado.",
      );
      setStatus("error");
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">BOE RAG</h1>
        <p className="app-subtitle">
          Respuestas <strong>citadas y verificables</strong> sobre legislación española,
          fundamentadas en el Boletín Oficial del Estado.
        </p>
      </header>

      <main className="app-main">
        <AskForm onAsk={handleAsk} loading={status === "loading"} />

        {asked && status !== "idle" && (
          <p className="asked-question">
            <span className="asked-label">Pregunta:</span> {asked}
          </p>
        )}

        {status === "loading" && (
          <div className="status loading" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            Buscando en el BOE y redactando una respuesta fundamentada…
          </div>
        )}

        {status === "error" && error && (
          <div className="status error" role="alert">
            {error}
          </div>
        )}

        {status === "done" && result && <AnswerPanel result={result} />}
      </main>

      <footer className="app-footer">
        <p>
          Demo de portafolio. Las respuestas pueden contener errores: verifica siempre
          con la fuente oficial enlazada. No constituye asesoramiento legal.
        </p>
      </footer>
    </div>
  );
}
