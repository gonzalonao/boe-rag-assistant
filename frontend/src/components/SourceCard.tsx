import type { Source } from "../types";

interface SourceCardProps {
  source: Source;
  index: number;
}

/** A single cited passage: its bracket number, citation, snippet, and BOE link. */
export function SourceCard({ source, index }: SourceCardProps) {
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
      </div>
      <p className="source-text">{source.text}</p>
    </li>
  );
}
