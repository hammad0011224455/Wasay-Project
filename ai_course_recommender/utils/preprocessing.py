"""Data preprocessing utilities."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_datasets(
    courses: pd.DataFrame,
    interactions: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions into train/validation/test while ensuring consistency."""
    interactions = interactions.dropna(subset=["user_id", "course_id", "rating"])
    train_val, test = train_test_split(
        interactions,
        test_size=test_size,
        random_state=random_state,
        stratify=interactions["user_id"],
    )
    train, val = train_test_split(
        train_val,
        test_size=test_size,
        random_state=random_state,
        stratify=train_val["user_id"],
    )
    valid_users = set(train["user_id"]).union(val["user_id"]).union(test["user_id"])
    valid_courses = set(courses["course_id"].astype(str))
    courses = courses[courses["course_id"].astype(str).isin(valid_courses)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


__all__ = ["prepare_datasets"]
