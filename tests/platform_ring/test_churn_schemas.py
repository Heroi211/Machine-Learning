"""Testes de schema — domínio churn."""

import pytest
from pydantic import ValidationError

from platform_ring.schemas.churn_features import ChurnFeaturesInput


class TestChurnFeaturesInput:
    def test_validates_telco_payload(self):
        payload = {
            "gender": "Female",
            "seniorcitizen": 0,
            "partner": 1,
            "dependents": 0,
            "tenure": 12,
            "phoneservice": 1,
            "multiplelines": 0,
            "internetservice": "DSL",
            "onlinesecurity": 0,
            "onlinebackup": 0,
            "deviceprotection": 0,
            "techsupport": 0,
            "streamingtv": 0,
            "streamingmovies": 0,
            "contract": "Month-to-month",
            "paperlessbilling": 1,
            "paymentmethod": "Electronic check",
            "monthlycharges": 70.5,
            "totalcharges": 845.0,
        }

        model = ChurnFeaturesInput.model_validate(payload)

        assert model.gender == "Female"
        assert model.tenure == 12

    def test_rejects_missing_required_field(self):
        with pytest.raises(ValidationError):
            ChurnFeaturesInput.model_validate({"gender": "Female"})

    def test_forbids_extra_fields(self):
        payload = {
            "gender": "Female",
            "seniorcitizen": 0,
            "partner": 1,
            "dependents": 0,
            "tenure": 12,
            "phoneservice": 1,
            "multiplelines": 0,
            "internetservice": "DSL",
            "onlinesecurity": 0,
            "onlinebackup": 0,
            "deviceprotection": 0,
            "techsupport": 0,
            "streamingtv": 0,
            "streamingmovies": 0,
            "contract": "Month-to-month",
            "paperlessbilling": 1,
            "paymentmethod": "Electronic check",
            "monthlycharges": 70.5,
            "totalcharges": 845.0,
            "extra": 1,
        }

        with pytest.raises(ValidationError):
            ChurnFeaturesInput.model_validate(payload)
