"""Testes do registry TrainBackend."""

from __future__ import annotations

import domains  # noqa: F401 — registra domínios e backends
from ml_core_ring.train_backend import TRAIN_BACKEND_REGISTRY, get_train_backend


def test_train_backend_registry_has_legacy_and_recommendation():
    assert "legacy_fe" in TRAIN_BACKEND_REGISTRY
    assert "recommendation_dvc" in TRAIN_BACKEND_REGISTRY
    assert get_train_backend("legacy_fe").backend_id == "legacy_fe"
    assert get_train_backend("recommendation_dvc").backend_id == "recommendation_dvc"
