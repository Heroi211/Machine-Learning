"""Worker leve de recomendação (Fase 5: container HTTP dedicado)."""

from __future__ import annotations

from typing import Any

from ml_core_ring.orchestration_hooks import run_training_for_domain
from ml_core_ring.train_backend import TrainBackendResult


def run_training_job(params: dict[str, Any] | None = None) -> TrainBackendResult:
    """Entrypoint partilhado por CLI, DVC worker e futuro serviço HTTP."""
    conf = dict(params or {})
    domain = str(conf.pop("domain", "recommendation")).strip().lower()
    return run_training_for_domain(domain, conf)


def result_to_dict(result: TrainBackendResult) -> dict[str, Any]:
    """Serialização simples para JSON / logs do worker."""
    payload: dict[str, Any] = {
        "status": result.status,
        "detail": result.detail,
        "metrics": result.metrics,
        "mlflow_run_id": result.mlflow_run_id,
    }
    if result.run_result is not None:
        payload["champion_name"] = result.run_result.champion_name
    return payload
