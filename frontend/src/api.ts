import type { AnswerResponse, SearchResponse } from "./types";

// Base URL of the JSON API. Injected at build time (VITE_API_BASE_URL = the
// deployed HF Space); empty string means same-origin (useful for local proxying).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/** An error carrying a user-facing, Spanish message ready to render. */
export class ApiError extends Error {}

const MESSAGES: Record<number, string> = {
  429: "Demasiadas peticiones seguidas. Espera unos segundos e inténtalo de nuevo.",
  503: "El servicio de generación no está disponible ahora mismo. Vuelve a intentarlo en un momento.",
};

/** POST a JSON body to an API endpoint, mapping failures to a typed ApiError. */
async function postJson<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "No se ha podido conectar con el servicio. Comprueba tu conexión e inténtalo de nuevo.",
    );
  }
  if (!response.ok) {
    throw new ApiError(
      MESSAGES[response.status] ?? `Error inesperado (${response.status}).`,
    );
  }
  return (await response.json()) as T;
}

/**
 * Ask the assistant a question and get a grounded, cited answer.
 *
 * @param question Natural-language question about Spanish legislation.
 * @param k Number of passages to ground the answer in (1–20).
 * @throws ApiError with a user-facing message on a non-2xx response or network failure.
 */
export function ask(question: string, k = 5): Promise<AnswerResponse> {
  return postJson<AnswerResponse>("/ask", { question, k });
}

/**
 * Retrieve the raw passages ranked for a query, without generating an answer.
 *
 * @param query The search query about Spanish legislation.
 * @param k Number of passages to return (1–20).
 * @throws ApiError with a user-facing message on a non-2xx response or network failure.
 */
export function search(query: string, k = 10): Promise<SearchResponse> {
  return postJson<SearchResponse>("/search", { query, k });
}
