import pytest
from fastapi.testclient import TestClient

from main import app
from api.v1.endpoints import processor
from models.predictions import Predictions
from services.processor.inference_report import build_inference_report


class DummyUser:
    id = 1


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    app.dependency_overrides[processor.get_current_user] = lambda: DummyUser()
    app.dependency_overrides[processor.get_session] = lambda: None

    async def fake_predict_for_domain(domain, features, user_id, db):
        pred = Predictions(
            id=1,
            user_id=user_id,
            pipeline_run_id=123,
            input_data=features,
            prediction=1,
            probability=0.8,
        )
        report = build_inference_report(
            {"inference_backend": "sklearn", "predict_model": "sklearn_pipeline"},
            "sklearn",
        )
        return pred, report

    monkeypatch.setattr(processor.processor_service, "predict_for_domain", fake_predict_for_domain)
    yield
    app.dependency_overrides.clear()


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
    assert response.json()["inference_report"]["predict_model"] == "sklearn_pipeline"


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
