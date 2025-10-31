"""Hybrid recommender combining content-based and collaborative models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HybridConfig:
    content_weight: float = 0.5
    collaborative_weight: float = 0.5

    def normalize(self) -> None:
        total = self.content_weight + self.collaborative_weight
        if total == 0:
            self.content_weight = 0.5
            self.collaborative_weight = 0.5
            total = 1.0
        self.content_weight /= total
        self.collaborative_weight /= total


class HybridRecommender:
    def __init__(self, content_model, collaborative_model, config: Optional[HybridConfig] = None) -> None:
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.config = config or HybridConfig()
        self.config.normalize()

    def recommend(
        self,
        user_id: str,
        seed_courses: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        content_scores = dict(self.content_model.recommend(seed_courses, top_k=top_k * 2))
        collab_scores = dict(self.collaborative_model.recommend(user_id, top_k=top_k * 2))
        all_items = set(content_scores) | set(collab_scores)
        blended: List[Tuple[str, float, Dict[str, float]]] = []
        for item in all_items:
            content_score = content_scores.get(item, 0.0)
            collab_score = collab_scores.get(item, 0.0)
            blended_score = (
                self.config.content_weight * content_score
                + self.config.collaborative_weight * collab_score
            )
            blended.append((item, blended_score, {"content": content_score, "collaborative": collab_score}))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:top_k]


__all__ = ["HybridRecommender", "HybridConfig"]
