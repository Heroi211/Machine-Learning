"""API HTTP do worker de recomendação (treino isolado + persistência BD)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import domains  # noqa: F401
import executors_ring  # noqa: F401
from executors_ring.recommendation.worker import result_to_dict, run_training_job
from executors_ring.recommendation.persist_run import persist_recommendation_run, run_async

logger = logging.getLogger(__name__)

app = FastAPI(title="worker-recommendation", version="1.0.0")


class TrainRequest(BaseModel):
    domain: str = Field(default="recommendation")
    user_id: int = Field(default=2, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)
    airflow_dag_run_id: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "worker-recommendation"}


@app.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    """Executa treino + grava ``pipeline_runs`` (mesmo contrato que task Airflow in-process)."""
    train_params = {**req.params, "domain": req.domain.strip().lower()}
    result = run_training_job(train_params)

    if result.status != "completed":
        raise HTTPException(
            status_code=500,
            detail=f"Treino não concluído (status={result.status!r}): {result.detail}",
        )

    run_id = run_async(
        persist_recommendation_run(
            user_id=req.user_id,
            result=result,
            airflow_dag_run_id=req.airflow_dag_run_id,
        )
    )
    payload = result_to_dict(result)
    payload["pipeline_run_id"] = run_id
    logger.info("Treino worker concluído | pipeline_run_id=%s", run_id)
    return payload
