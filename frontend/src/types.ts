// Mirrors the FastAPI Pydantic models (src/boe_rag/service/models.py). The API
// is the single source of truth; keep these in sync with AnswerResponse/Source.

export interface Source {
  chunk_id: string;
  citation: string;
  text: string;
  url: string;
  score: number;
}

export interface AnswerResponse {
  answer: string;
  refused: boolean;
  sources: Source[];
}
