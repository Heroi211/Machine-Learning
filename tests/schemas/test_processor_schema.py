import pandas as pd
import pandera.pandas as pa
import pytest
from pydantic import ValidationError

from schemas.processor_schemas import HeartDiseaseFeaturesInput, MLDomain, PredictRequest


class TestProcessorSchemas:
    def test_predict_request_validates_pydantic_schema(self):
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

        request = PredictRequest.model_validate(payload)

        assert request.domain == MLDomain.heart_disease
        assert isinstance(request.features, HeartDiseaseFeaturesInput)
        assert request.features.age == 60.0
        assert request.features.cp_typical_angina is True

    def test_predict_request_rejects_missing_required_feature(self):
        payload = {
            "domain": "heart_disease",
            "features": {
                "age": 60.0,
                "trestbps": 130.0,
            },
        }

        with pytest.raises(ValidationError):
            PredictRequest.model_validate(payload)


class TestPredictionFeaturesDataFrameSchema:
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(float, nullable=False),
            "trestbps": pa.Column(float, nullable=False),
            "chol": pa.Column(float, nullable=False),
            "fbs": pa.Column(bool, nullable=False),
            "thalch": pa.Column(float, nullable=False),
            "exang": pa.Column(bool, nullable=False),
            "oldpeak": pa.Column(float, nullable=False),
            "ca": pa.Column(float, nullable=False),
            "sex_Male": pa.Column(bool, nullable=False),
            "cp_atypical angina": pa.Column(bool, nullable=False),
            "cp_non-anginal": pa.Column(bool, nullable=False),
            "cp_typical angina": pa.Column(bool, nullable=False),
            "restecg_normal": pa.Column(bool, nullable=False),
            "restecg_st-t abnormality": pa.Column(bool, nullable=False),
            "slope_flat": pa.Column(bool, nullable=False),
            "slope_upsloping": pa.Column(bool, nullable=False),
            "thal_normal": pa.Column(bool, nullable=False),
            "thal_reversable defect": pa.Column(bool, nullable=False),
        },
        strict=True,
        coerce=True,
    )

    def test_features_dataframe_validates_with_pandera(self):
        df = pd.DataFrame(
            [
                {
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
                }
            ]
        )

        validated = self.schema.validate(df)

        assert validated.shape == (1, 18)
        assert validated.loc[0, "age"] == 60.0
        assert bool(validated.loc[0, "thal_normal"]) is True
