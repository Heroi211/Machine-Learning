"""Facade pipeline DVC de recomendação."""

from domains.recommendation.pipeline_runner import (
    RecommendationPipelineRunner,
    build_run_context,
)

__all__ = ["RecommendationPipelineRunner", "build_run_context"]
