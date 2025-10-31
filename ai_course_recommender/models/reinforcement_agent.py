"""Contextual bandit for online personalization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BanditConfig:
    epsilon: float = 0.1
    min_visits: int = 1


@dataclass
class CourseStats:
    total_reward: float = 0.0
    visits: int = 0

    @property
    def average_reward(self) -> float:
        return self.total_reward / max(1, self.visits)


@dataclass
class ContextualBanditAgent:
    config: BanditConfig = field(default_factory=BanditConfig)
    course_stats: Dict[str, CourseStats] = field(default_factory=dict)

    def select_action(self, candidate_courses: List[str]) -> str:
        unexplored = [cid for cid in candidate_courses if self.course_stats.get(cid, CourseStats()).visits < self.config.min_visits]
        if unexplored:
            return np.random.choice(unexplored)
        if np.random.rand() < self.config.epsilon:
            return np.random.choice(candidate_courses)
        scores = {cid: self.course_stats[cid].average_reward for cid in candidate_courses}
        return max(scores, key=scores.get)

    def update(self, course_id: str, reward: float) -> None:
        stats = self.course_stats.setdefault(course_id, CourseStats())
        stats.total_reward += reward
        stats.visits += 1

    def recommend(self, candidate_courses: List[str], num_recommendations: int = 5) -> List[Tuple[str, float]]:
        ranked: List[Tuple[str, float]] = []
        for _ in range(min(num_recommendations, len(candidate_courses))):
            course_id = self.select_action(candidate_courses)
            avg_reward = self.course_stats.get(course_id, CourseStats()).average_reward
            ranked.append((course_id, avg_reward))
        seen = set()
        unique_ranked: List[Tuple[str, float]] = []
        for cid, score in ranked:
            if cid not in seen:
                unique_ranked.append((cid, score))
                seen.add(cid)
        return unique_ranked


__all__ = ["ContextualBanditAgent", "BanditConfig"]
