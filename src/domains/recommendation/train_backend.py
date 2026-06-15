"""Backend de treino recomendação — pipeline DVC / PipelineRunner."""

from __future__ import annotations

from ml_core_ring.train_backend import TrainBackendResult, TrainRequestContext, register_train_backend


class RecommendationDvcTrainBackend:
    backend_id = "recommendation_dvc"

    def run_train(self, ctx: TrainRequestContext) -> TrainBackendResult:
        from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context

        run_ctx = build_run_context({**ctx.params, "domain": ctx.domain})
        result = RecommendationPipelineRunner().run(run_ctx)
        return TrainBackendResult(
            status="completed",
            detail=f"Champion: {result.champion_name}",
            metrics=result.metrics,
            mlflow_run_id=result.mlflow_run_id,
            run_result=result,
        )


register_train_backend(RecommendationDvcTrainBackend())
