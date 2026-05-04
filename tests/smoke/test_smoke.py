"""
Smoke tests rápidos cobrindo os principais fluxos da aplicação.
Estes testes verificam se endpoints críticos funcionam de ponta a ponta.
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from api.v1.endpoints import processor, authorize
from models.predictions import Predictions
from services.auth import auth_service


class DummyUser:
    """Usuário simulado para testes."""

    def __init__(self, id=1, role_id=1):
        self.id = id
        self.role_id = role_id


class MockRegisteredUser:
    """Usuário cadastrado simulado para testes de autenticação."""

    def __init__(self, id, name, email, active=True, role_id=1):
        self.id = id
        self.name = name
        self.email = email
        self.active = active
        self.role_id = role_id


@pytest.fixture(autouse=True)
def app_overrides(monkeypatch):
    """Substitui dependências da aplicação para todos os smoke tests."""
    # Substitui dependências do processor.
    app.dependency_overrides[processor.get_current_user] = lambda: DummyUser()
    app.dependency_overrides[processor.get_session] = lambda: None

    # Substitui dependências de autenticação.
    app.dependency_overrides[authorize.get_session] = lambda: None

    # Simula serviço de predição.
    async def fake_predict_for_domain(domain, features, user_id, db):
        return Predictions(
            id=1,
            user_id=user_id,
            pipeline_run_id=123,
            input_data=features,
            prediction=1,
            probability=0.8,
        )

    # Simula serviço de autenticação.
    async def fake_register_user(user, db):
        return MockRegisteredUser(id=1, name=user.name, email=user.email)

    monkeypatch.setattr(processor.processor_service, "predict_for_domain", fake_predict_for_domain)
    monkeypatch.setattr(auth_service, "register_user", fake_register_user)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """Cria cliente de teste para a aplicação FastAPI."""
    return TestClient(app)


class TestSmokeAuthFlow:
    """Smoke tests do fluxo de autenticação."""

    def test_signup_endpoint_works(self, client):
        """Verifica se o endpoint de cadastro de usuário está funcionando."""
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
    """Smoke tests do fluxo de predição de ML."""

    def test_predict_endpoint_works(self, client):
        """Verifica se o endpoint de predição está funcionando."""
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
