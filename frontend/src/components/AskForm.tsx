import { type FormEvent, useState } from "react";

interface AskFormProps {
  onAsk: (question: string) => void;
  loading: boolean;
  placeholder?: string;
  label?: string;
  loadingLabel?: string;
}

// Every example must be answerable from the served corpus (BOE Disposiciones
// Generales, 2015–present). The first two come from the gold eval set (scored
// 1.00/1.00 end-to-end); the third was verified against the live /search index.
const EXAMPLES = [
  "¿Dónde tiene su sede principal el Instituto Vasco de Finanzas?",
  "¿Qué empresa asume las obligaciones de servicio público para el voto por correo en las elecciones de 2024?",
  "¿Qué regula la Ley 39/2015?",
];

/** The query input with a submit button and a few example prompts.
 *
 * The button/placeholder copy is configurable so the same form serves both the
 * "ask" (grounded answer) and "search" (raw retrieval) modes.
 */
export function AskForm({
  onAsk,
  loading,
  placeholder = "Pregunta sobre legislación española…",
  label = "Preguntar",
  loadingLabel = "Consultando…",
}: AskFormProps) {
  const [value, setValue] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !loading) {
      onAsk(trimmed);
    }
  }

  return (
    <form className="ask-form" onSubmit={submit}>
      <div className="ask-row">
        <input
          className="ask-input"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          maxLength={1000}
          disabled={loading}
        />
        <button
          className="ask-button"
          type="submit"
          disabled={loading || !value.trim()}
        >
          {loading ? loadingLabel : label}
        </button>
      </div>
      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="example-chip"
            onClick={() => {
              setValue(example);
              onAsk(example);
            }}
            disabled={loading}
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}
