"""Contratos compartilhados de ML (inferência, manifest, domínios)."""

from core.ml import engines  # noqa: F401 — registra ENGINE_REGISTRY
from core.ml.artifact_manifest import ArtifactManifest
from core.ml.inference_engine import ENGINE_REGISTRY, InferenceEngine, get_engine
from core.ml.prediction_result import PredictionResult

__all__ = [
    "ArtifactManifest",
    "InferenceEngine",
    "PredictionResult",
    "ENGINE_REGISTRY",
    "get_engine",
    "engines",
]
