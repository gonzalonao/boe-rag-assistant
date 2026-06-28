import type { AnswerResponse } from "../types";
import { SourceCard } from "./SourceCard";

interface AnswerPanelProps {
  result: AnswerResponse;
}

/** Renders a grounded answer with its cited sources, or a clear refusal notice. */
export function AnswerPanel({ result }: AnswerPanelProps) {
  if (result.refused) {
    return (
      <section className="answer-panel refused" aria-live="polite">
        <p className="refusal">
          No he encontrado base suficiente en el BOE para responder a esa pregunta, así
          que prefiero no inventar. Prueba a reformularla o a preguntar por otra norma.
        </p>
      </section>
    );
  }

  return (
    <section className="answer-panel" aria-live="polite">
      <p className="answer-text">{result.answer}</p>
      {result.sources.length > 0 && (
        <>
          <h2 className="sources-title">Fuentes citadas</h2>
          <ol className="sources-list">
            {result.sources.map((source, i) => (
              <SourceCard key={source.chunk_id} source={source} index={i + 1} />
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
