"""
Persistência partilhada Airflow → ``pipeline_runs``.

Tabular (baseline/FE): delega a ``services.processor.airflow_persistence`` (transição Fase 4).
Recomendação: ``persist_recommendation_run`` neste módulo.
"""

from __future__ import annotations

import logging
from typing import Any

from core.database import Session
import models._all_models  # noqa: F401
from models.pipeline_runs import PipelineRuns
from ml_core_ring.train_backend import TrainBackendResult
from services.utils import utcnow

logger = logging.getLogger(__name__)

# Re-export tabular (legado platform — migrar na Fase 4)
from services.processor.airflow_persistence import (  # noqa: E402
    deactivate_manual_pipeline_runs_for_objective,
    persist_airflow_baseline_run,
    persist_airflow_feature_engineering_run,
    promote_airflow_fe_if_requested,
    reserve_airflow_fe_pipeline_run,
    run_async,
)

__all__ = [
    "deactivate_manual_pipeline_runs_for_objective",
    "persist_airflow_baseline_run",
    "persist_airflow_feature_engineering_run",
    "persist_recommendation_run",
    "promote_airflow_fe_if_requested",
    "reserve_airflow_fe_pipeline_run",
    "run_async",
]


async def persist_recommendation_run(
    *,
    user_id: int,
    result: TrainBackendResult,
    airflow_dag_run_id: str | None = None,
) -> int:
    """Grava run de recomendação concluído (pipeline_type=recommendation)."""
    if result.run_result is None:
        raise ValueError("TrainBackendResult.run_result é obrigatório para persistência de recomendação.")

    manifest = result.run_result.manifest
    metrics: dict[str, Any] = dict(result.metrics or {})
    metrics["champion_name"] = result.run_result.champion_name
    metrics["domain"] = manifest.domain
    metrics["problem_type"] = manifest.problem_type
    metrics["manifest_engine"] = manifest.engine
    if manifest.artifacts:
        metrics["artifact_paths"] = dict(manifest.artifacts)
    if result.mlflow_run_id:
        metrics["mlflow_run_id"] = result.mlflow_run_id
    if airflow_dag_run_id:
        metrics["airflow_dag_run_id"] = airflow_dag_run_id
    metrics["airflow_dag"] = "ml_training_dispatch"

    model_path = manifest.artifacts.get("prefix") or manifest.artifacts.get("joblib_path")
    inference_backend = "sklearn"
    if "torch" in str(manifest.engine).lower():
        inference_backend = "mlp"

    session = Session()
    try:
        run = PipelineRuns(
            user_id=int(user_id),
            pipeline_type="recommendation",
            objective=manifest.domain,
            status="completed",
            original_filename=str(metrics.get("data_source", "recommendation_pipeline")),
            model_path=str(model_path) if model_path else None,
            metrics=metrics,
            completed_at=utcnow(),
            is_airflow_run=True,
            inference_backend=inference_backend,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        logger.info("Recomendação Airflow guardada (pipeline_run_id=%s).", run.id)
        return run.id
    finally:
        await session.close()
