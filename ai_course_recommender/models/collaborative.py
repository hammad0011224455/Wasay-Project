"""Collaborative filtering models including matrix factorization and neural approaches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class MFConfig:
    n_components: int = 50
    random_state: int = 42


@dataclass
class NCFConfig:
    n_factors: int = 32
    hidden_dim: int = 64
    n_epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cpu"


class MatrixFactorizationRecommender:
    """Matrix Factorization using Truncated SVD on the user-course matrix."""

    def __init__(self, config: Optional[MFConfig] = None) -> None:
        self.config = config or MFConfig()
        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        pivot = interactions.pivot_table(index="user_id", columns="course_id", values="rating", fill_value=0.0)
        self.user_index = {uid: idx for idx, uid in enumerate(pivot.index.astype(str))}
        self.item_index = {cid: idx for idx, cid in enumerate(pivot.columns.astype(str))}
        self.svd = TruncatedSVD(n_components=self.config.n_components, random_state=self.config.random_state)
        latent = self.svd.fit_transform(pivot.values)
        self.user_factors = latent
        self.item_factors = self.svd.components_.T

    def recommend(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model not fitted")
        if user_id not in self.user_index:
            raise ValueError(f"Unknown user_id: {user_id}")
        user_vector = self.user_factors[self.user_index[user_id]]
        scores = np.dot(self.item_factors, user_vector)
        ranked_items = np.argsort(scores)[::-1]
        recommendations: List[Tuple[str, float]] = []
        items = list(self.item_index.keys())
        for idx in ranked_items:
            course_id = items[idx]
            recommendations.append((course_id, float(scores[idx])))
            if len(recommendations) >= top_k:
                break
        return recommendations


class InteractionDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, user_map: Dict[str, int], item_map: Dict[str, int]):
        self.user_tensor = torch.tensor([user_map[uid] for uid in interactions["user_id"].astype(str)], dtype=torch.long)
        self.item_tensor = torch.tensor([item_map[cid] for cid in interactions["course_id"].astype(str)], dtype=torch.long)
        self.rating_tensor = torch.tensor(interactions["rating"].astype(np.float32).values)

    def __len__(self) -> int:
        return len(self.rating_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_tensor[idx], self.item_tensor[idx], self.rating_tensor[idx]


class NeuralCFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, config: NCFConfig) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, config.n_factors)
        self.item_embedding = nn.Embedding(n_items, config.n_factors)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_factors * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        features = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(features).squeeze(-1)


class NeuralCollaborativeFiltering:
    """Neural Collaborative Filtering using PyTorch."""

    def __init__(self, config: Optional[NCFConfig] = None) -> None:
        self.config = config or NCFConfig()
        self.model: Optional[NeuralCFModel] = None
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        interactions = interactions.copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["course_id"] = interactions["course_id"].astype(str)
        self.user_map = {uid: idx for idx, uid in enumerate(interactions["user_id"].unique())}
        self.item_map = {cid: idx for idx, cid in enumerate(interactions["course_id"].unique())}

        dataset = InteractionDataset(interactions, self.user_map, self.item_map)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model = NeuralCFModel(len(self.user_map), len(self.item_map), self.config).to(self.config.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.model.train()
        for _ in range(self.config.n_epochs):
            for user_batch, item_batch, rating_batch in loader:
                user_batch = user_batch.to(self.config.device)
                item_batch = item_batch.to(self.config.device)
                rating_batch = rating_batch.to(self.config.device)
                optimizer.zero_grad()
                preds = self.model(user_batch, item_batch)
                loss = criterion(preds, rating_batch)
                loss.backward()
                optimizer.step()

    def recommend(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        if user_id not in self.user_map:
            raise ValueError(f"Unknown user_id: {user_id}")
        self.model.eval()
        user_idx = torch.tensor([self.user_map[user_id]], device=self.config.device)
        scores: List[Tuple[str, float]] = []
        for course_id, item_idx in self.item_map.items():
            item_tensor = torch.tensor([item_idx], device=self.config.device)
            with torch.no_grad():
                pred = self.model(user_idx, item_tensor).item()
            scores.append((course_id, float(pred)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


__all__ = [
    "MatrixFactorizationRecommender",
    "NeuralCollaborativeFiltering",
    "MFConfig",
    "NCFConfig",
]
