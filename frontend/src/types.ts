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

export interface SearchResponse {
  query: string;
  results: Source[];
}

// --- Quality page (src/quality-data.json, built from reports/*.json) ---

/** A confidence interval (bootstrap) for a retrieval metric. */
export interface MetricCI {
  low: number;
  high: number;
}

/** Headline retrieval metrics for one eval set (gold or silver). */
export interface RetrievalStats {
  k: number;
  num_queries: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  recall_ci: MetricCI | null;
  mrr_ci: MetricCI | null;
}

/** One pipeline stage in the retrieval ablation. */
export interface AblationRow {
  name: string;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
}

/** End-to-end answer-quality metrics (LLM-judged). */
export interface GenerationStats {
  num_queries: number;
  mean_faithfulness: number;
  mean_correctness: number;
  refusal_rate: number;
}

/** Adversarial pass rate for one attack category. */
export interface SecurityCategory {
  name: string;
  pass_rate: number;
}

/** Aggregate adversarial-eval posture. */
export interface SecurityStats {
  num_cases: number;
  num_passed: number;
  pass_rate: number;
  by_category: SecurityCategory[];
}

/** The full curated quality dataset rendered by the quality page. */
export interface QualityData {
  generated: string;
  retrieval: {
    gold: RetrievalStats;
    silver: RetrievalStats;
  };
  ablation: {
    k: number;
    rows: AblationRow[];
  };
  generation: GenerationStats;
  security: SecurityStats;
}
