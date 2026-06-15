"""Registry de motores de inferência plugáveis."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from core.ml.artifact_manifest import ArtifactManifest
from core.ml.prediction_result import PredictionResult


class InferenceEngine(Protocol):
    backend_id: str

    def load(self, manifest: ArtifactManifest) -> None: ...

    def predict(self, df_input: pd.DataFrame) -> PredictionResult: ...


ENGINE_REGISTRY: dict[str, type[InferenceEngine]] = {}


def register_engine(engine_cls: type[InferenceEngine]) -> type[InferenceEngine]:
    ENGINE_REGISTRY[engine_cls.backend_id] = engine_cls
    return engine_cls


def get_engine(manifest: ArtifactManifest) -> InferenceEngine:
    engine_cls = ENGINE_REGISTRY.get(manifest.engine)
    if engine_cls is None:
        known = ", ".join(sorted(ENGINE_REGISTRY)) or "(vazio)"
        raise ValueError(f"Engine {manifest.engine!r} não registrado. Disponíveis: {known}")
    engine = engine_cls()
    engine.load(manifest)
    return engine
