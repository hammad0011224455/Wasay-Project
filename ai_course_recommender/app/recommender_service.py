"""Service layer orchestrating the recommender pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from ..models.content_based import ContentBasedRecommender
from ..models.collaborative import MatrixFactorizationRecommender
from ..models.hybrid_model import HybridRecommender
from ..models.reinforcement_agent import ContextualBanditAgent
from ..utils.explainability import build_explanation


@dataclass
class UserProfile:
    user_id: str
    level: str
    interest: str
    recent_courses: List[str]


class RecommenderService:
    def __init__(
        self,
        courses: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        self.courses = courses
        self.interactions = interactions
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = MatrixFactorizationRecommender()
        self.bandit_agent = ContextualBanditAgent()
        self.hybrid_model: HybridRecommender | None = None

    def train_models(self) -> None:
        self.content_model.fit(self.courses)
        self.collaborative_model.fit(self.interactions)
        self.hybrid_model = HybridRecommender(self.content_model, self.collaborative_model)

    def get_recommendations(self, profile: UserProfile, top_k: int = 5) -> List[Dict[str, str]]:
        if self.hybrid_model is None:
            raise RuntimeError("Models must be trained before getting recommendations")
        hybrid_results = self.hybrid_model.recommend(profile.user_id, profile.recent_courses, top_k=top_k * 2)
        candidate_courses = [course_id for course_id, *_ in hybrid_results]
        bandit_ranked = self.bandit_agent.recommend(candidate_courses, num_recommendations=top_k)
        final_results: List[Dict[str, str]] = []
        for course_id, _ in bandit_ranked:
            hybrid_scores = next(scores for cid, _, scores in hybrid_results if cid == course_id)
            explanation = build_explanation(course_id, profile.recent_courses, hybrid_scores, {
                "level": profile.level,
                "interest": profile.interest,
            })
            final_results.append({
                "course_id": course_id,
                "title": self._course_title(course_id),
                "explanation": explanation,
            })
        return final_results

    def update_feedback(self, course_id: str, reward: float) -> None:
        self.bandit_agent.update(course_id, reward)

    def _course_title(self, course_id: str) -> str:
        row = self.courses[self.courses["course_id"].astype(str) == str(course_id)]
        if row.empty:
            return str(course_id)
        return row.iloc[0]["title"]


__all__ = ["RecommenderService", "UserProfile"]
