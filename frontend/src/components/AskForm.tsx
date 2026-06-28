import { type FormEvent, useState } from "react";

interface AskFormProps {
  onAsk: (question: string) => void;
  loading: boolean;
}

const EXAMPLES = [
  "¿Cuál es el tipo general del IVA?",
  "¿Qué plazo hay para recurrir una sanción de tráfico?",
  "¿Qué regula la Ley 39/2015?",
];

/** The question input with a submit button and a few example prompts. */
export function AskForm({ onAsk, loading }: AskFormProps) {
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
          placeholder="Pregunta sobre legislación española…"
          aria-label="Pregunta sobre legislación española"
          maxLength={1000}
          disabled={loading}
        />
        <button
          className="ask-button"
          type="submit"
          disabled={loading || !value.trim()}
        >
          {loading ? "Consultando…" : "Preguntar"}
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
