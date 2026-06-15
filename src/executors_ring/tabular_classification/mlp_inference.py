"""Inferência MLP tabular — facade para o bundle PyTorch gravado no FE."""

from services.pipelines.mlp_inference import (
    MlpBundle,
    load_mlp_bundle,
    predict_with_mlp,
)

__all__ = ["MlpBundle", "load_mlp_bundle", "predict_with_mlp"]
