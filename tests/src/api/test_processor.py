import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.v1.endpoints import processor


class DummyUser:
    id = 1


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    app.dependency_overrides[processor.get_current_user] = lambda: DummyUser()
    app.dependency_overrides[processor.get_session] = lambda: None


def test_predict_endpoint_returns_success(client):
    payload = {
        "domain": "heart_disease",
        "features": {
            "age": 60.0,
            "trestbps": 130.0,
            "chol": 250.0,
            "fbs": False,
            "thalch": 150.0,
            "exang": False,
            "oldpeak": 1.5,
            "ca": 0.0,
            "sex_Male": True,
            "cp_atypical angina": False,
            "cp_non-anginal": False,
            "cp_typical angina": True,
            "restecg_normal": True,
            "restecg_st-t abnormality": False,
            "slope_flat": False,
            "slope_upsloping": True,
            "thal_normal": True,
            "thal_reversable defect": False,
        },
    }

    response = client.post("/v1/processor/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probability"] == 80.0
    assert response.json()["probability_display"] == "80.0%"
    assert (
        response.json()["inference_report"]["served_model"]["predict_model_key"]
        == "sklearn_pipeline"
    )


def test_predict_endpoint_invalid_payload_returns_422(client):
    payload = {
        "domain": "heart_disease",
        "features": {"age": 60.0}
    }

    response = client.post("/v1/processor/predict", json=payload)

    assert response.status_code == 422


@pytest.fixture
def client():
    return TestClient(app)
