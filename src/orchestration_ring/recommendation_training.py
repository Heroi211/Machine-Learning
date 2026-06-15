"""Tasks Airflow — treino de recomendação via registry + persist_run."""

from __future__ import annotations

import logging

import domains  # noqa: F401
import executors_ring  # noqa: F401
from ml_core_ring.orchestration_hooks import run_training_for_domain

from orchestration_ring.airflow_env import DEFAULT_PIPELINE_USER_ID
from orchestration_ring.conf import merge_run_conf
from orchestration_ring.dispatch import resolve_domain
from orchestration_ring.persist_run import persist_recommendation_run, run_async

logger = logging.getLogger(__name__)


def task_validate_recommendation_input(**context) -> None:
    """Valida conf mínima para recomendação (sem CSV tabular)."""
    conf = merge_run_conf(context, variable_key="ml_training_dispatch_conf")
    domain = resolve_domain(conf)
    ti = context["task_instance"]
    ti.xcom_push(key="domain", value=domain)
    ti.xcom_push(key="user_id", value=int(conf.get("user_id", DEFAULT_PIPELINE_USER_ID)))
    for key in ("top_k", "train_models", "mlflow_experiment", "n_epochs"):
        if conf.get(key) is not None:
            ti.xcom_push(key=key, value=conf.get(key))
    logger.info("Conf recomendação validada | domain=%s", domain)


def task_run_recommendation(**context) -> None:
    """Executa pipeline DVC/recomendação e persiste ``pipeline_runs``."""
    ti = context["task_instance"]
    domain = ti.xcom_pull(key="domain", task_ids="validate_dispatch")
    user_id = int(ti.xcom_pull(key="user_id", task_ids="validate_dispatch") or DEFAULT_PIPELINE_USER_ID)

    conf = merge_run_conf(context, variable_key="ml_training_dispatch_conf")
    train_params = {
        k: v
        for k, v in conf.items()
        if k not in ("domain", "objective", "user_id", "csv_path")
    }

    logger.info("Treino recomendação | domain=%s | params=%s", domain, list(train_params.keys()))
    result = run_training_for_domain(domain, train_params)

    if result.status != "completed":
        raise RuntimeError(
            f"Treino recomendação não concluído (status={result.status!r}): {result.detail}"
        )

    run_id = run_async(
        persist_recommendation_run(
            user_id=user_id,
            result=result,
            airflow_dag_run_id=context["dag_run"].run_id,
        )
    )

    ti.xcom_push(key="recommendation_pipeline_run_id", value=run_id)
    ti.xcom_push(key="recommendation_metrics", value=result.metrics)
    ti.xcom_push(key="recommendation_champion", value=result.run_result.champion_name if result.run_result else None)
    ti.xcom_push(key="mlflow_run_id", value=result.mlflow_run_id)
    logger.info(
        "Recomendação concluída | pipeline_run_id=%s | champion=%s",
        run_id,
        result.run_result.champion_name if result.run_result else None,
    )


def task_notify_recommendation_complete(**context) -> None:
    ti = context["task_instance"]
    domain = ti.xcom_pull(key="domain", task_ids="validate_dispatch")
    run_id = ti.xcom_pull(key="recommendation_pipeline_run_id", task_ids="run_recommendation")
    champion = ti.xcom_pull(key="recommendation_champion", task_ids="run_recommendation")
    metrics = ti.xcom_pull(key="recommendation_metrics", task_ids="run_recommendation")

    logger.info("=" * 60)
    logger.info("PIPELINE RECOMENDAÇÃO CONCLUÍDO | domain=%s", domain)
    logger.info("pipeline_run_id : %s", run_id)
    logger.info("campeão         : %s", champion)
    logger.info("métricas        : %s", metrics)
    logger.info("=" * 60)
