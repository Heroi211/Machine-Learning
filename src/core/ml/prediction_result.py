"""Resultado unificado de inferência (classificação ou recomendação)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class PredictionResult:
    """Saída normalizada de um ``InferenceEngine``."""

    problem_type: Literal["binary_classification", "recommendation"]
    label: int | None = None
    probability: float | None = None
    item_ids: list[int] | None = None
    scores: list[float] | None = None
    raw: dict[str, Any] | None = None
