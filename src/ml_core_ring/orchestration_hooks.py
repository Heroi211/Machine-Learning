"""Pontes entre orquestradores (Airflow) e backends de treino."""

from __future__ import annotations

from typing import Any

from ml_core_ring.domain_plugin import get_domain
from ml_core_ring.train_backend import TrainBackendResult, TrainRequestContext, get_train_backend


def run_training_for_domain(domain: str, conf: dict[str, Any] | None = None) -> TrainBackendResult:
    """Resolve domínio → backend e executa treino conforme registry."""
    plugin = get_domain(domain)
    backend = get_train_backend(plugin.pipeline_runner_id)
    ctx = TrainRequestContext(
        domain=plugin.name,
        params=dict(conf or {}),
        triggered_by="orchestration",
    )
    return backend.run_train(ctx)
