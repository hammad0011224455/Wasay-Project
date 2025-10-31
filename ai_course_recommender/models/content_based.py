"""Content-based recommendation models using TF-IDF and BERT embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


@dataclass
class ContentBasedConfig:
    tfidf_max_features: int = 5000
    bert_model_name: str = "all-MiniLM-L6-v2"
    use_bert: bool = True


class ContentBasedRecommender:
    """Content-based recommender that supports TF-IDF and BERT embeddings."""

    def __init__(self, config: Optional[ContentBasedConfig] = None) -> None:
        self.config = config or ContentBasedConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.bert_model: Optional[SentenceTransformer] = None
        self.bert_embeddings: Optional[np.ndarray] = None
        self.course_ids: List[str] = []

    def fit(self, courses: pd.DataFrame) -> None:
        """Fits both TF-IDF and BERT models on course metadata."""
        if "course_id" not in courses.columns:
            raise ValueError("courses dataframe must include a 'course_id' column")
        self.course_ids = courses["course_id"].astype(str).tolist()
        corpus = (
            courses["title"].fillna("").astype(str)
            + " "
            + courses["description"].fillna("").astype(str)
        )

        self.vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray()

        if self.config.use_bert and SentenceTransformer is not None:
            self.bert_model = SentenceTransformer(self.config.bert_model_name)
            self.bert_embeddings = self.bert_model.encode(corpus, show_progress_bar=False)

    def _get_embedding_matrix(self) -> np.ndarray:
        if self.config.use_bert and self.bert_embeddings is not None:
            return self.bert_embeddings
        if self.tfidf_matrix is not None:
            return self.tfidf_matrix
        raise RuntimeError("Model is not fitted yet")

    def recommend(self, seed_course_ids: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Recommend courses similar to the ones provided."""
        if not seed_course_ids:
            raise ValueError("seed_course_ids cannot be empty")
        matrix = self._get_embedding_matrix()
        seed_indices = [self.course_ids.index(str(cid)) for cid in seed_course_ids if str(cid) in self.course_ids]
        if not seed_indices:
            raise ValueError("None of the seed courses are known to the model")
        seed_vector = np.mean(matrix[seed_indices], axis=0)
        similarities = cosine_similarity(seed_vector.reshape(1, -1), matrix).flatten()
        ranked_indices = np.argsort(similarities)[::-1]
        recommendations: List[Tuple[str, float]] = []
        for idx in ranked_indices:
            course_id = self.course_ids[idx]
            if course_id in seed_course_ids:
                continue
            recommendations.append((course_id, float(similarities[idx])))
            if len(recommendations) >= top_k:
                break
        return recommendations


__all__ = ["ContentBasedRecommender", "ContentBasedConfig"]
