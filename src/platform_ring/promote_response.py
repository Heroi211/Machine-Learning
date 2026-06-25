"""Constrói resposta HTTP de promote com metadados MLflow Registry (Fase 6)."""

from __future__ import annotations

from models.deployed_models import DeployedModels
from platform_ring.mlflow_registry import RegistryPromoteResult
from schemas import processor_schemas


def build_deployed_model_response(
    deployment: DeployedModels,
    registry: RegistryPromoteResult | None = None,
) -> processor_schemas.DeployedModelResponse:
    run = deployment.pipeline_run
    pipeline_type = run.pipeline_type if run else None
    warning = None
    model_name = None
    version = None
    stage = None
    run_id = None

    if registry is not None:
        if registry.skipped:
            warning = registry.skip_reason
        elif registry.warning:
            warning = registry.warning
            model_name = registry.model_name or None
        elif registry.version:
            model_name = registry.model_name
            version = registry.version
            stage = registry.stage
            run_id = registry.mlflow_run_id

    return processor_schemas.DeployedModelResponse(
        id=deployment.id,
        domain=deployment.domain,
        pipeline_run_id=deployment.pipeline_run_id,
        status=deployment.status,
        promoted_at=deployment.promoted_at,
        promoted_by_user_id=deployment.promoted_by_user_id,
        metrics_snapshot=deployment.metrics_snapshot,
        mlflow_registry_model=model_name,
        mlflow_registry_version=version,
        mlflow_registry_stage=stage,
        mlflow_registry_run_id=run_id,
        mlflow_registry_warning=warning,
        pipeline_type=pipeline_type,
    )
