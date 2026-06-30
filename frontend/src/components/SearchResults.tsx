import type { SearchResponse } from "../types";
import { SourceCard } from "./SourceCard";

interface SearchResultsProps {
  result: SearchResponse;
}

/** Renders the ranked passages returned by raw retrieval (no generation). */
export function SearchResults({ result }: SearchResultsProps) {
  if (result.results.length === 0) {
    return (
      <section className="answer-panel" aria-live="polite">
        <p className="refusal">
          No se han encontrado pasajes para esa búsqueda. Prueba con otros términos.
        </p>
      </section>
    );
  }

  return (
    <section className="answer-panel" aria-live="polite">
      <h2 className="sources-title">{result.results.length} pasajes recuperados</h2>
      <ol className="sources-list">
        {result.results.map((source, i) => (
          <SourceCard key={source.chunk_id} source={source} index={i + 1} showScore />
        ))}
      </ol>
    </section>
  );
}
