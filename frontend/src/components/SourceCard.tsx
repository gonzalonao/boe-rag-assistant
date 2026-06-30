import type { Source } from "../types";

interface SourceCardProps {
  source: Source;
  index: number;
  showScore?: boolean;
}

/** A single cited passage: its bracket number, citation, snippet, and BOE link.
 *
 * In search mode the retrieval score is surfaced as a badge; in answer mode it
 * is hidden, since the bracket number there refers to the in-text citation.
 */
export function SourceCard({ source, index, showScore = false }: SourceCardProps) {
  return (
    <li className="source-card">
      <div className="source-head">
        <span className="source-index">[{index}]</span>
        <a
          className="source-citation"
          href={source.url}
          target="_blank"
          rel="noreferrer"
        >
          {source.citation}
        </a>
        {showScore && (
          <span className="source-score" title="Puntuación de relevancia">
            {source.score.toFixed(3)}
          </span>
        )}
      </div>
      <p className="source-text">{source.text}</p>
    </li>
  );
}
