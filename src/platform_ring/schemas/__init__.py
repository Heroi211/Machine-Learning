"""Contratos HTTP da platform_ring."""

from platform_ring.schemas.churn_features import ChurnFeaturesInput
from platform_ring.schemas.contracts import (
    BaselinePredictBlock,
    ComparisonPredict,
    DeployedModelResponse,
    ExperimentationPredict,
    InferenceReport,
    MetricSnapshot,
    PipelineRunResponse,
    PredictResponse,
    PyTorchMLPExperiment,
    ServedModelPredict,
    TrainingSelectionSummaryPredict,
    TriggerDagResponse,
)
from platform_ring.schemas.recommendation_features import RecommendationFeaturesInput

__all__ = [
    "BaselinePredictBlock",
    "ChurnFeaturesInput",
    "ComparisonPredict",
    "DeployedModelResponse",
    "ExperimentationPredict",
    "InferenceReport",
    "MetricSnapshot",
    "PipelineRunResponse",
    "PredictResponse",
    "PyTorchMLPExperiment",
    "RecommendationFeaturesInput",
    "ServedModelPredict",
    "TrainingSelectionSummaryPredict",
    "TriggerDagResponse",
]
