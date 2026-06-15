"""Testes manifest e promote recomendação."""

from __future__ import annotations

from dataclasses import dataclass

import domains  # noqa: F401
from ml_core_ring.artifact_manifest import ArtifactManifest


@dataclass
class FakeRecoRun:
    objective: str = "recommendation"
    inference_backend: str = "mlp"
    model_path: str = "/models/reco/torch_embedding"
    pipeline_type: str = "recommendation"
    metrics: dict | None = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {
                "problem_type": "recommendation",
                "manifest_engine": "torch_embedding",
                "champion_name": "torch_embedding",
                "top_k": 10,
            }


def test_artifact_manifest_from_recommendation_run():
    manifest = ArtifactManifest.from_pipeline_run(FakeRecoRun())
    assert manifest.problem_type == "recommendation"
    assert manifest.engine == "recommendation_torch"
    assert manifest.artifacts["prefix"] == "/models/reco/torch_embedding"
