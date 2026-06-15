"""Plugin de domínio recomendação (Tech Challenge Fase 02)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ml_core_ring.domain_plugin import DomainPlugin, register_domain


class RecommendationInput(BaseModel):
    """Entrada mínima para recomendação user-item."""

    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(..., ge=1, description="Identificador do utilizador")
    top_k: int = Field(default=10, ge=1, le=100, description="Número de itens a recomendar")


register_domain(
    DomainPlugin(
        name="recommendation",
        problem_type="recommendation",
        feature_strategy=None,
        input_schema=RecommendationInput,
        class_labels=None,
        allowed_metrics=frozenset(
            {"hit_rate", "precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"}
        ),
        pipeline_runner_id="recommendation_dvc",
    )
)
