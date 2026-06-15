"""Backend de treino churn — delegação ao pipeline Airflow legado (Baseline + FE)."""

from __future__ import annotations

from ml_core_ring.train_backend import TrainBackendResult, TrainRequestContext, register_train_backend


class LegacyFeTrainBackend:
    backend_id = "legacy_fe"

    def run_train(self, ctx: TrainRequestContext) -> TrainBackendResult:
        return TrainBackendResult(
            status="orchestrated",
            detail=(
                "Treino tabular churn executado pelo DAG Airflow ml_training_pipeline "
                "(Baseline + Feature Engineering). Fase 2 moverá lógica para executors_ring."
            ),
        )


register_train_backend(LegacyFeTrainBackend())
