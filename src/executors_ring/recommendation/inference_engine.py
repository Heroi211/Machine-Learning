"""Engine de inferência recomendação (embedding PyTorch)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.inference_engine import register_engine
from ml_core_ring.paths import resolve_shared_artifact_path
from ml_core_ring.prediction_result import PredictionResult


@register_engine
class RecommendationTorchEngine:
    backend_id = "recommendation_torch"

    def __init__(self) -> None:
        self._recommender = None
        self._default_top_k = 10

    def load(self, manifest: ArtifactManifest) -> None:
        prefix = manifest.artifacts.get("prefix")
        if not prefix:
            raise ValueError("Manifest recommendation_torch sem artifacts['prefix'].")
        resolved = resolve_shared_artifact_path(prefix)
        if not resolved:
            raise ValueError(f"Prefix inválido no manifest: {prefix!r}")
        from domains.recommendation.models.embedding_model import TorchEmbeddingRecommender

        self._recommender = TorchEmbeddingRecommender.load(Path(resolved))
        self._default_top_k = int(manifest.metadata.get("top_k", 10))

    def predict(self, df_input: pd.DataFrame) -> PredictionResult:
        if self._recommender is None:
            raise RuntimeError("Engine recommendation_torch não carregado.")
        row = df_input.iloc[0]
        user_id = int(row["user_id"])
        top_k = int(row.get("top_k", self._default_top_k))
        items = self._recommender.recommend(user_id, top_k)
        return PredictionResult(
            problem_type="recommendation",
            item_ids=[int(i) for i in items],
            raw={"user_id": user_id, "top_k": top_k},
        )
