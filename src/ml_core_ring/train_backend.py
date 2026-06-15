"""Registry de backends de treino (Airflow, CLI, workers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ml_core_ring.run_context import RunResult


@dataclass
class TrainRequestContext:
    domain: str
    params: dict[str, Any] = field(default_factory=dict)
    triggered_by: str | None = None
    dag_run_id: str | None = None


@dataclass
class TrainBackendResult:
    status: str
    detail: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    mlflow_run_id: str | None = None
    run_result: RunResult | None = None


class TrainBackend(Protocol):
    backend_id: str

    def run_train(self, ctx: TrainRequestContext) -> TrainBackendResult: ...


TRAIN_BACKEND_REGISTRY: dict[str, TrainBackend] = {}


def register_train_backend(backend: TrainBackend) -> TrainBackend:
    TRAIN_BACKEND_REGISTRY[backend.backend_id] = backend
    return backend


def get_train_backend(backend_id: str) -> TrainBackend:
    backend = TRAIN_BACKEND_REGISTRY.get(backend_id)
    if backend is None:
        known = ", ".join(sorted(TRAIN_BACKEND_REGISTRY)) or "(vazio)"
        raise KeyError(f"TrainBackend {backend_id!r} não registrado. Disponíveis: {known}")
    return backend
