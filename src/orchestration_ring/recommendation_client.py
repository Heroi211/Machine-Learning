"""Cliente HTTP — Airflow → worker de recomendação."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def run_recommendation_via_worker(
    *,
    worker_url: str,
    domain: str,
    user_id: int,
    params: dict[str, Any],
    airflow_dag_run_id: str | None = None,
    timeout_seconds: float = 7200.0,
) -> dict[str, Any]:
    """POST /train no worker integrado; devolve JSON com pipeline_run_id e métricas."""
    base = worker_url.rstrip("/")
    payload = {
        "domain": domain,
        "user_id": user_id,
        "params": params,
        "airflow_dag_run_id": airflow_dag_run_id,
    }
    logger.info("Delegando treino reco ao worker %s | domain=%s", base, domain)
    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(f"{base}/train", json=payload)
        resp.raise_for_status()
        return resp.json()
