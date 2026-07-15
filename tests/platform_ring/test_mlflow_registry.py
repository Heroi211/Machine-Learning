"""Testes Fase 6 — MLflow Registry side-effect no promote."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from platform_ring.mlflow_registry import (
    RegistryPromoteResult,
    _promote_registry_version,
    resolve_registry_model_name,
    sync_mlflow_registry_on_promote,
)
from platform_ring.promote_response import build_deployed_model_response


def test_resolve_registry_model_name_recommendation():
    assert resolve_registry_model_name("recommendation") == "tc02_recommender"


def test_resolve_registry_model_name_unknown():
    assert resolve_registry_model_name("fraud") is None


def test_sync_skips_unknown_domain():
    result = sync_mlflow_registry_on_promote(
        domain="unknown",
        metrics={"mlflow_run_id": "abc"},
        pipeline_type="recommendation",
    )
    assert result is not None
    assert result.skipped is True


@patch("mlflow.tracking.set_tracking_uri")
@patch("mlflow.MlflowClient")
def test_sync_promotes_version_via_alias(mock_client_cls, mock_set_uri):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.set_registered_model_alias = MagicMock()
    del client.transition_model_version  # MLflow 3.x

    mv = MagicMock()
    mv.version = "3"
    mv.run_id = "run-123"
    client.search_model_versions.return_value = [mv]

    result = sync_mlflow_registry_on_promote(
        domain="recommendation",
        metrics={"mlflow_run_id": "run-123"},
        pipeline_type="recommendation",
    )

    assert result is not None
    assert result.version == "3"
    assert result.stage == "Production"
    assert result.model_name == "tc02_recommender"
    client.set_registered_model_alias.assert_called_once_with("tc02_recommender", "Production", "3")
    mock_set_uri.assert_called_once()


def test_promote_registry_version_legacy_stages():
    client = MagicMock(spec=["transition_model_version"])
    client.transition_model_version = MagicMock()

    _promote_registry_version(client, model_name="tc02_recommender", version="3")

    assert client.transition_model_version.call_count == 2


@patch("mlflow.tracking.set_tracking_uri")
@patch("mlflow.MlflowClient")
def test_sync_best_effort_on_registry_failure(mock_client_cls, mock_set_uri):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.set_registered_model_alias = MagicMock(side_effect=RuntimeError("MLflow offline"))

    mv = MagicMock()
    mv.version = "1"
    mv.run_id = "run-1"
    client.search_model_versions.return_value = [mv]

    result = sync_mlflow_registry_on_promote(
        domain="recommendation",
        metrics={"mlflow_run_id": "run-1"},
        pipeline_type="recommendation",
    )

    assert result is not None
    assert result.warning is not None
    assert "MLflow offline" in result.warning


def test_build_deployed_model_response_with_registry():
    deployment = MagicMock()
    deployment.id = 1
    deployment.domain = "recommendation"
    deployment.pipeline_run_id = 10
    deployment.status = "active"
    deployment.promoted_at = None
    deployment.promoted_by_user_id = 2
    deployment.metrics_snapshot = {"ndcg_at_k": 0.5}
    deployment.pipeline_run = MagicMock(pipeline_type="recommendation")

    registry = RegistryPromoteResult(
        model_name="tc02_recommender",
        version="2",
        stage="Production",
        mlflow_run_id="abc",
    )
    resp = build_deployed_model_response(deployment, registry)

    assert resp.mlflow_registry_model == "tc02_recommender"
    assert resp.mlflow_registry_version == "2"
    assert resp.mlflow_registry_stage == "Production"
    assert resp.pipeline_type == "recommendation"
