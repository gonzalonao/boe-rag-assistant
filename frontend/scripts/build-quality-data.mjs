// Assemble the quality page's data from the committed eval reports.
//
// The SPA is a static GitHub Pages site with no backend, so it cannot read the
// repo's reports/ at runtime. This script (run via the `prebuild`/`predev` npm
// hooks) curates the headline numbers from reports/*.json into a single typed
// JSON module that the page imports at build time. reports/ stays the single
// source of truth; the deployed page always reflects the latest committed eval.
//
// It fails loudly if a required report or field is missing, so a blank metrics
// page can never ship silently.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPORTS_DIR = resolve(HERE, "..", "..", "reports");
const OUT_FILE = resolve(HERE, "..", "src", "quality-data.json");

/** Read and parse a report JSON, failing with a clear message if absent. */
function readReport(name) {
  const path = resolve(REPORTS_DIR, name);
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw new Error(`Cannot read required report ${name} at ${path}: ${err.message}`);
  }
}

/** Assert a field exists on a report object (numbers may legitimately be 0). */
function field(report, key, where) {
  const value = report[key];
  if (value === undefined || value === null) {
    throw new Error(`Report ${where} is missing required field "${key}".`);
  }
  return value;
}

/** Project a retrieval report into the page's gold/silver stats shape. */
function retrievalStats(report, where) {
  const recallCi = report.recall_at_k_ci;
  const mrrCi = report.mrr_ci;
  return {
    k: field(report, "k", where),
    num_queries: field(report, "num_queries", where),
    recall_at_k: field(report, "recall_at_k", where),
    mrr: field(report, "mrr", where),
    ndcg_at_k: field(report, "ndcg_at_k", where),
    recall_ci: recallCi ? { low: recallCi.low, high: recallCi.high } : null,
    mrr_ci: mrrCi ? { low: mrrCi.low, high: mrrCi.high } : null,
  };
}

// The reranker ablation report keys, mapped to human-readable pipeline stages.
const ABLATION_STAGES = [
  ["dense", "Denso (e5)"],
  ["hybrid (RRF)", "Híbrido (RRF)"],
  ["hybrid + cross-encoder", "Híbrido + cross-encoder"],
];

function ablation(report) {
  const rows = ABLATION_STAGES.map(([key, name]) => {
    const stage = report[key];
    if (!stage) {
      throw new Error(`retrieval_rerank.json is missing the "${key}" stage.`);
    }
    return {
      name,
      recall_at_k: field(stage, "recall_at_k", `retrieval_rerank.${key}`),
      mrr: field(stage, "mrr", `retrieval_rerank.${key}`),
      ndcg_at_k: field(stage, "ndcg_at_k", `retrieval_rerank.${key}`),
    };
  });
  const k = report.dense ? field(report.dense, "k", "retrieval_rerank.dense") : 10;
  return { k, rows };
}

function security(report) {
  const byCategory = field(report, "pass_rate_by_category", "security_eval");
  return {
    num_cases: field(report, "num_cases", "security_eval"),
    num_passed: field(report, "num_passed", "security_eval"),
    pass_rate: field(report, "pass_rate", "security_eval"),
    by_category: Object.entries(byCategory)
      .map(([name, pass_rate]) => ({ name, pass_rate }))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
}

const gold = readReport("retrieval_baseline.json");
const silver = readReport("evalset_silver_retrieval.json");
const rerank = readReport("retrieval_rerank.json");
const e2e = readReport("e2e_baseline.json");
const sec = readReport("security_eval.json");

const data = {
  generated: new Date().toISOString(),
  retrieval: {
    gold: retrievalStats(gold, "retrieval_baseline"),
    silver: retrievalStats(silver, "evalset_silver_retrieval"),
  },
  ablation: ablation(rerank),
  generation: {
    num_queries: field(e2e, "num_queries", "e2e_baseline"),
    mean_faithfulness: field(e2e, "mean_faithfulness", "e2e_baseline"),
    mean_correctness: field(e2e, "mean_correctness", "e2e_baseline"),
    refusal_rate: field(e2e, "refusal_rate", "e2e_baseline"),
  },
  security: security(sec),
};

writeFileSync(OUT_FILE, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(
  `Wrote ${OUT_FILE} (${data.security.num_cases} security cases, ` +
    `${data.retrieval.silver.num_queries} silver queries).`,
);
