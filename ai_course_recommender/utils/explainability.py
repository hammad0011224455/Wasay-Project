"""Explainability utilities for recommendations."""
from __future__ import annotations

from typing import Dict, List


def build_explanation(
    recommended_course: str,
    seed_courses: List[str],
    hybrid_scores: Dict[str, float],
    user_profile: Dict[str, str],
) -> str:
    liked_course = seed_courses[0] if seed_courses else "your previous learning"
    difficulty = user_profile.get("level", "unspecified level")
    focus_area = user_profile.get("interest", "similar topics")
    content_score = hybrid_scores.get("content", 0.0)
    collab_score = hybrid_scores.get("collaborative", 0.0)
    return (
        f"Recommended '{recommended_course}' because you liked '{liked_course}', "
        f"have shown interest in {focus_area}, and similar learners at the {difficulty} "
        f"level achieved scores content={content_score:.2f}, collaborative={collab_score:.2f}."
    )


__all__ = ["build_explanation"]
