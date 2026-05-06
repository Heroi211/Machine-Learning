"""
Smoke Tests - Quick integration tests covering main application flows.
These tests verify that critical endpoints work end-to-end.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.v1.endpoints import processor, authorize
from src.services.auth import auth_service
from src.services.processor.inference_report import build_inference_report


class DummyUser:
    """Mock user for testing"""
    def __init__(self, id=1, role_id=1):
        self.id = id
        self.role_id = role_id


class MockRegisteredUser:
    """Mock registered user for auth tests"""
    def __init__(self, id, name, email, active=True, role_id=1):
        self.id = id
        self.name = name
        self.email = email
        self.active = active
        self.role_id = role_id


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    """Override app dependencies for all smoke tests"""
    # Override processor dependencies
    app.dependency_overrides[processor.get_current_user] = lambda: DummyUser()
    app.dependency_overrides[processor.get_session] = lambda: None

    # Override auth dependencies
    app.dependency_overrides[authorize.get_session] = lambda: None

    # Mock auth service
    async def fake_register_user(user, db):
        return MockRegisteredUser(id=1, name=user.name, email=user.email)

    monkeypatch.setattr(auth_service, "register_user", fake_register_user)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestSmokeAuthFlow:
    """Smoke tests for authentication flow"""

    def test_signup_endpoint_works(self, client):
        """Test that user signup endpoint is working"""
        payload = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpass123",
        }

        response = client.post("/v1/auth/signup", json=payload)

        assert response.status_code == 201
        assert response.json()["id"] == 1
        assert response.json()["email"] == "test@example.com"


class TestSmokePredictorFlow:
    """Smoke tests for ML predictor flow"""

    def test_predict_endpoint_works(self, client):
        """Test that prediction endpoint is working"""
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
        body = response.json()
        assert "inference_report" in body
        assert body["inference_report"]["inference_backend"] == "sklearn"
