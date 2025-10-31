"""Evaluation metrics and simulation tools."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def precision_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / max(1, k)


def recall_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / max(1, len(relevant))


def ndcg_at_k(recommended: Sequence[str], relevant: Sequence[str], k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    discounts = [1.0 / np.log2(idx + 2) for idx in range(len(gains))]
    dcg = sum(g * d for g, d in zip(gains, discounts))
    ideal_gains = sorted(gains, reverse=True)
    ideal_dcg = sum(g * d for g, d in zip(ideal_gains, discounts))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def mean_reciprocal_rank(recommended: Sequence[str], relevant: Sequence[str]) -> float:
    for idx, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def evaluate_ranking(recommended: Sequence[str], relevant: Sequence[str], k: int = 5) -> Dict[str, float]:
    return {
        "precision@k": precision_at_k(recommended, relevant, k),
        "recall@k": recall_at_k(recommended, relevant, k),
        "ndcg@k": ndcg_at_k(recommended, relevant, k),
        "mrr": mean_reciprocal_rank(recommended, relevant),
    }


def ab_test_simulation(model_a_scores: Iterable[float], model_b_scores: Iterable[float]) -> Dict[str, float]:
    model_a_scores = list(model_a_scores)
    model_b_scores = list(model_b_scores)
    lift = np.mean(model_b_scores) - np.mean(model_a_scores)
    return {
        "model_a_mean": float(np.mean(model_a_scores)),
        "model_b_mean": float(np.mean(model_b_scores)),
        "lift": float(lift),
    }


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_reciprocal_rank",
    "evaluate_ranking",
    "ab_test_simulation",
]
