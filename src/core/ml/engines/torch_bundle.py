"""Adapter bundle PyTorch (MLP tabular) para o registry de inferência."""

from __future__ import annotations

import pandas as pd

from core.ml.artifact_manifest import ArtifactManifest
from core.ml.inference_engine import register_engine
from core.ml.paths import resolve_shared_artifact_path
from core.ml.prediction_result import PredictionResult


@register_engine
class TorchBundleEngine:
    backend_id = "torch_bundle"

    def __init__(self) -> None:
        self._bundle = None

    def load(self, manifest: ArtifactManifest) -> None:
        from services.pipelines.mlp_inference import load_mlp_bundle

        prefix = manifest.artifacts.get("prefix")
        if not prefix:
            raise ValueError("Manifest torch_bundle sem artifacts['prefix'].")
        resolved = resolve_shared_artifact_path(prefix)
        if not resolved:
            raise ValueError("Prefix MLP inválido no manifest.")
        self._bundle = load_mlp_bundle(resolved)

    def predict(self, df_input: pd.DataFrame) -> PredictionResult:
        if self._bundle is None:
            raise RuntimeError("Engine torch_bundle não carregado.")
        from services.pipelines.mlp_inference import predict_with_mlp

        label, prob = predict_with_mlp(self._bundle, df_input)
        return PredictionResult(
            problem_type="binary_classification",
            label=int(label),
            probability=float(prob),
        )
