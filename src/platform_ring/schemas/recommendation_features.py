"""Payload de predict — domínio recommendation."""

from pydantic import BaseModel, ConfigDict, Field


class RecommendationFeaturesInput(BaseModel):
    """Entrada mínima para recomendação user-item."""

    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(..., ge=1, description="Identificador do utilizador")
    top_k: int = Field(default=10, ge=1, le=100, description="Número de itens a recomendar")
