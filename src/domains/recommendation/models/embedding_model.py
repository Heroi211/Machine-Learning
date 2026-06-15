"""Modelo de recomendação embedding-based em PyTorch (TC02)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _EmbeddingDotModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(dim=1)


class TorchEmbeddingRecommender:
    name = "torch_embedding"

    def __init__(
        self,
        embedding_dim: int = 32,
        n_epochs: int = 10,
        batch_size: int = 512,
        lr: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._reverse_items: dict[int, int] = {}
        self._model: _EmbeddingDotModel | None = None
        self._user_items: dict[int, set[int]] = {}
        self._popular_items: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        torch.manual_seed(self.random_state)
        users = sorted(train_df["user_id"].astype(int).unique().tolist())
        items = sorted(train_df["item_id"].astype(int).unique().tolist())
        self._user_map = {u: i for i, u in enumerate(users)}
        self._item_map = {it: i for i, it in enumerate(items)}
        self._reverse_items = {i: it for it, i in self._item_map.items()}
        self._user_items = train_df.groupby("user_id")["item_id"].apply(lambda s: set(map(int, s))).to_dict()
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self._popular_items = [int(i) for i in counts.index.tolist()]

        u_idx = train_df["user_id"].astype(int).map(self._user_map).to_numpy(dtype=np.int64)
        i_idx = train_df["item_id"].astype(int).map(self._item_map).to_numpy(dtype=np.int64)
        y = train_df["rating"].astype(np.float32).to_numpy()

        dataset = TensorDataset(
            torch.from_numpy(u_idx),
            torch.from_numpy(i_idx),
            torch.from_numpy(y),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = _EmbeddingDotModel(len(users), len(items), self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.n_epochs):
            for batch_u, batch_i, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_u, batch_i)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
        self._model = model

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]:
        exclude = exclude_items or set()
        seen = self._user_items.get(user_id, set()) | exclude
        if self._model is None or user_id not in self._user_map:
            pop = PopularityFallback(self._popular_items)
            return pop.recommend(user_id, n_items, exclude_items=seen)

        self._model.eval()
        user_idx = self._user_map[user_id]
        item_indices = torch.arange(len(self._item_map), dtype=torch.long)
        user_tensor = torch.full((len(item_indices),), user_idx, dtype=torch.long)
        with torch.no_grad():
            scores = self._model(user_tensor, item_indices).numpy()
        order = np.argsort(-scores)
        out: list[int] = []
        for idx in order:
            item = self._reverse_items[int(idx)]
            if item in seen:
                continue
            out.append(item)
            if len(out) >= n_items:
                break
        return out

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("Modelo torch não treinado.")
        torch.save(self._model.state_dict(), prefix.with_suffix(".pt"))
        meta = {
            "embedding_dim": self.embedding_dim,
            "user_map": self._user_map,
            "item_map": self._item_map,
            "popular_items": self._popular_items,
        }
        prefix.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> TorchEmbeddingRecommender:
        prefix = Path(prefix)
        meta = json.loads(prefix.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(embedding_dim=int(meta["embedding_dim"]))
        obj._user_map = {int(k): int(v) for k, v in meta["user_map"].items()}
        obj._item_map = {int(k): int(v) for k, v in meta["item_map"].items()}
        obj._reverse_items = {i: it for it, i in obj._item_map.items()}
        obj._popular_items = [int(x) for x in meta["popular_items"]]
        model = _EmbeddingDotModel(len(obj._user_map), len(obj._item_map), obj.embedding_dim)
        model.load_state_dict(torch.load(prefix.with_suffix(".pt"), map_location="cpu"))
        model.eval()
        obj._model = model
        return obj


class PopularityFallback:
    def __init__(self, popular_items: list[int]) -> None:
        self._popular_items = popular_items

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]:
        exclude = exclude_items or set()
        out: list[int] = []
        for item in self._popular_items:
            if item in exclude:
                continue
            out.append(item)
            if len(out) >= n_items:
                break
        return out
