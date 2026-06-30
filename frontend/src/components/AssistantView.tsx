import { useState } from "react";
import { ApiError, ask, search } from "../api";
import type { AnswerResponse, SearchResponse } from "../types";
import { AnswerPanel } from "./AnswerPanel";
import { AskForm } from "./AskForm";
import { SearchResults } from "./SearchResults";

type Status = "idle" | "loading" | "done" | "error";
type Mode = "ask" | "search";

const LOADING_COPY: Record<Mode, string> = {
  ask: "Buscando en el BOE y redactando una respuesta fundamentada…",
  search: "Recuperando los pasajes más relevantes del BOE…",
};

/** The interactive tool: ask for a grounded answer, or search raw passages. */
export function AssistantView() {
  const [mode, setMode] = useState<Mode>("ask");
  const [status, setStatus] = useState<Status>("idle");
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState<string | null>(null);

  function switchMode(next: Mode) {
    if (next === mode) {
      return;
    }
    setMode(next);
    setStatus("idle");
    setAnswer(null);
    setResults(null);
    setError(null);
    setAsked(null);
  }

  async function handleSubmit(query: string) {
    setStatus("loading");
    setError(null);
    setAnswer(null);
    setResults(null);
    setAsked(query);
    try {
      if (mode === "ask") {
        setAnswer(await ask(query));
      } else {
        setResults(await search(query));
      }
      setStatus("done");
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Ha ocurrido un error inesperado.",
      );
      setStatus("error");
    }
  }

  return (
    <>
      <div className="mode-toggle" role="tablist" aria-label="Modo de consulta">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "ask"}
          className={`mode-tab ${mode === "ask" ? "active" : ""}`}
          onClick={() => switchMode("ask")}
        >
          Preguntar
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "search"}
          className={`mode-tab ${mode === "search" ? "active" : ""}`}
          onClick={() => switchMode("search")}
        >
          Buscar pasajes
        </button>
      </div>

      <AskForm
        onAsk={handleSubmit}
        loading={status === "loading"}
        placeholder={
          mode === "ask"
            ? "Pregunta sobre legislación española…"
            : "Busca términos en la legislación española…"
        }
        label={mode === "ask" ? "Preguntar" : "Buscar"}
      />

      {asked && status !== "idle" && (
        <p className="asked-question">
          <span className="asked-label">
            {mode === "ask" ? "Pregunta:" : "Búsqueda:"}
          </span>{" "}
          {asked}
        </p>
      )}

      {status === "loading" && (
        <div className="status loading" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          {LOADING_COPY[mode]}
        </div>
      )}

      {status === "error" && error && (
        <div className="status error" role="alert">
          {error}
        </div>
      )}

      {status === "done" && mode === "ask" && answer && <AnswerPanel result={answer} />}
      {status === "done" && mode === "search" && results && (
        <SearchResults result={results} />
      )}
    </>
  );
}
