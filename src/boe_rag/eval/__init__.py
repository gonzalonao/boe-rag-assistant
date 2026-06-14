"""Evaluation harness: ranking metrics, golden datasets, and eval runners.

Built before any retrieval optimization so that every later change to the
pipeline must demonstrate a measured improvement against a fixed scorecard.
"""

from boe_rag.eval.dataset import EvalExample, load_evalset, save_evalset
from boe_rag.eval.metrics import RetrievalMetrics, evaluate_retrieval

__all__ = [
    "EvalExample",
    "RetrievalMetrics",
    "evaluate_retrieval",
    "load_evalset",
    "save_evalset",
]
