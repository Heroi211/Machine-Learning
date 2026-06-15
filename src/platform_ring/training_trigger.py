"""Disparo genérico de treino via Airflow DAG ``ml_training_dispatch``."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from core.configs import settings
from ml_core_ring.domain_plugin import get_domain

DISPATCH_DAG_ID = "ml_training_dispatch"


@dataclass(frozen=True)
class TriggerDagResult:
    dag_run_id: str
    dag_id: str
    domain: str
    conf: dict[str, Any]
    csv_path: str | None
    message: str


def build_dispatch_conf(
    *,
    domain: str,
    user_id: int,
    csv_path: str | None = None,
    optimization_metric: str = "recall",
    min_precision: float | None = None,
    min_roc_auc: float | None = None,
    tuning_n_iter: int | None = None,
    time_limit_minutes: int = 30,
    acc_target: float | None = None,
    auto_promote: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta conf para ``ml_training_dispatch`` conforme ``DomainPlugin``."""
    d = domain.strip().lower()
    plugin = get_domain(d)
    conf: dict[str, Any] = {
        "domain": d,
        "user_id": int(user_id),
        **(extra or {}),
    }

    if plugin.problem_type == "binary_classification":
        if not csv_path:
            raise ValueError(f"Domínio tabular {d!r} exige ficheiro CSV (csv_path).")
        conf.update(
            {
                "objective": d,
                "csv_path": csv_path,
                "optimization_metric": optimization_metric,
                "time_limit_minutes": time_limit_minutes,
                "auto_promote": auto_promote,
            }
        )
        if min_precision is not None:
            conf["min_precision"] = min_precision
        if min_roc_auc is not None:
            conf["min_roc_auc"] = min_roc_auc
        if tuning_n_iter is not None:
            conf["tuning_n_iter"] = tuning_n_iter
        if acc_target is not None:
            conf["acc_target"] = acc_target
    elif plugin.problem_type == "recommendation":
        conf.setdefault("top_k", 10)
    else:
        raise ValueError(f"problem_type {plugin.problem_type!r} sem conf de treino definida.")

    return conf


async def trigger_training_dag(
    *,
    domain: str,
    user_id: int,
    csv_path: str | None = None,
    **kwargs: Any,
) -> TriggerDagResult:
    """Dispara ``ml_training_dispatch`` no Airflow REST API."""
    conf = build_dispatch_conf(domain=domain, user_id=user_id, csv_path=csv_path, **kwargs)
    d = domain.strip().lower()
    dag_run_id = f"manual__{d}_{uuid.uuid4().hex[:8]}"

    base_url = settings.airflow_base_url.rstrip("/")
    auth = (settings.airflow_user, settings.airflow_password)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base_url}/api/v1/dags/{DISPATCH_DAG_ID}/dagRuns",
            json={"dag_run_id": dag_run_id, "conf": conf},
            auth=auth,
        )
        resp.raise_for_status()

    csv = conf.get("csv_path")
    return TriggerDagResult(
        dag_run_id=dag_run_id,
        dag_id=DISPATCH_DAG_ID,
        domain=d,
        conf=conf,
        csv_path=str(csv) if csv else None,
        message=f"DAG {DISPATCH_DAG_ID} disparado. UI: {base_url}/dags/{DISPATCH_DAG_ID}/grid",
    )
