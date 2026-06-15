"""Plugin de domínio churn (Fase 01 — delegação legada)."""

from __future__ import annotations

from core.ml.domain_plugin import DomainPlugin, register_domain
from schemas.processor_schemas import ChurnFeaturesInput
from services.pipelines.feature_strategies.churn_features import ChurnFeatures

register_domain(
    DomainPlugin(
        name="churn",
        problem_type="binary_classification",
        feature_strategy=ChurnFeatures,
        input_schema=ChurnFeaturesInput,
        class_labels=("Não Churn", "Churn"),
        allowed_metrics=frozenset({"accuracy", "precision", "recall", "f1", "roc_auc"}),
        pipeline_runner_id="legacy_fe",
    )
)
