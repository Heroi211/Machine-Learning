"""Testes cliente worker recomendação."""

from unittest.mock import MagicMock, patch

from orchestration_ring.recommendation_client import run_recommendation_via_worker


@patch("orchestration_ring.recommendation_client.httpx.Client")
def test_run_recommendation_via_worker(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "pipeline_run_id": 99,
        "status": "completed",
        "champion_name": "torch_embedding",
        "metrics": {"ndcg_at_k": 0.5},
    }
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client_cls.return_value = mock_client

    payload = run_recommendation_via_worker(
        worker_url="http://worker_recommendation:8010",
        domain="recommendation",
        user_id=2,
        params={"top_k": 5},
        airflow_dag_run_id="dag-1",
    )

    assert payload["pipeline_run_id"] == 99
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0].endswith("/train")
