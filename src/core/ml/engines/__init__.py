"""Motores de inferência registrados (import side-effect)."""

from core.ml.engines import sklearn_joblib, torch_bundle

__all__ = ["sklearn_joblib", "torch_bundle"]
