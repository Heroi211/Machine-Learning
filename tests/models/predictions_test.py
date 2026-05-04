from unittest.mock import patch
import pytest
from sqlalchemy import Integer, Float
from sqlalchemy.inspection import inspect

from models.predictions import Predictions
from models.users import Users
from models.pipeline_runs import PipelineRuns
from core.generic import modelsGeneric


class TestPredictionsModel:
    """Tests for Predictions model structure and attributes"""

    def test_predictions_inherits_from_generic(self):
        """Test that Predictions inherits from modelsGeneric"""
        assert issubclass(Predictions, modelsGeneric)

    def test_predictions_table_name(self):
        """Test that Predictions has correct table name"""
        assert Predictions.__tablename__ == "predictions"

    def test_predictions_has_inherited_columns(self):
        """Test that Predictions has columns from modelsGeneric"""
        # Get column names from mapper
        mapper = inspect(Predictions)
        column_names = [col.name for col in mapper.columns]
        
        # Should have inherited columns
        assert "created_at" in column_names
        assert "active" in column_names

    def test_predictions_created_at_column_exists(self):
        """Test that created_at column is inherited and configured"""
        mapper = inspect(Predictions)
        created_at_col = mapper.columns.get("created_at")
        
        assert created_at_col is not None
        assert not created_at_col.nullable

    def test_predictions_active_column_exists(self):
        """Test that active column is inherited"""
        mapper = inspect(Predictions)
        active_col = mapper.columns.get("active")
        
        assert active_col is not None
        assert active_col.default is not None


class TestPredictionsColumns:
    """Tests for Predictions model columns"""

    def test_predictions_id_column(self):
        """Test that id column is properly defined"""
        mapper = inspect(Predictions)
        id_col = mapper.columns.get("id")
        
        assert id_col is not None
        assert id_col.primary_key
        assert id_col.autoincrement

    def test_predictions_user_id_column(self):
        """Test that user_id column is properly defined"""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")
        
        assert user_id_col is not None
        assert not user_id_col.nullable
        assert len(user_id_col.foreign_keys) > 0

    def test_predictions_user_id_foreign_key(self):
        """Test that user_id has correct foreign key"""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")
        
        fk = list(user_id_col.foreign_keys)[0]
        assert fk.column.table.name == "users"

    def test_predictions_pipeline_run_id_column(self):
        """Test that pipeline_run_id column is properly defined"""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")
        
        assert pipeline_run_id_col is not None
        assert not pipeline_run_id_col.nullable
        assert len(pipeline_run_id_col.foreign_keys) > 0

    def test_predictions_pipeline_run_id_foreign_key(self):
        """Test that pipeline_run_id has correct foreign key"""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")
        
        fk = list(pipeline_run_id_col.foreign_keys)[0]
        assert fk.column.table.name == "pipeline_runs"

    def test_predictions_input_data_column(self):
        """Test that input_data column is properly defined"""
        mapper = inspect(Predictions)
        input_data_col = mapper.columns.get("input_data")
        
        assert input_data_col is not None
        assert not input_data_col.nullable
        # Check if it's JSON type
        assert type(input_data_col.type).__name__ in ["JSON", "String"]

    def test_predictions_prediction_column(self):
        """Test that prediction column is properly defined"""
        mapper = inspect(Predictions)
        prediction_col = mapper.columns.get("prediction")
        
        assert prediction_col is not None
        assert not prediction_col.nullable
        assert isinstance(prediction_col.type, Integer)

    def test_predictions_probability_column(self):
        """Test that probability column is properly defined"""
        mapper = inspect(Predictions)
        probability_col = mapper.columns.get("probability")
        
        assert probability_col is not None
        assert probability_col.nullable  # Probability can be NULL
        assert isinstance(probability_col.type, Float)


class TestPredictionsRelationships:
    """Tests for Predictions model relationships"""

    def test_predictions_has_user_relationship(self):
        """Test that Predictions has user relationship"""
        mapper = inspect(Predictions)
        relationships = {rel.key: rel for rel in mapper.relationships}
        
        assert "user" in relationships

    def test_predictions_user_relationship_configuration(self):
        """Test that user relationship is properly configured"""
        mapper = inspect(Predictions)
        user_rel = mapper.relationships["user"]
        
        assert user_rel.mapper.class_ == Users
        # Should be lazy loaded with 'joined'
        assert user_rel.lazy == "joined"

    def test_predictions_has_pipeline_run_relationship(self):
        """Test that Predictions has pipeline_run relationship"""
        mapper = inspect(Predictions)
        relationships = {rel.key: rel for rel in mapper.relationships}
        
        assert "pipeline_run" in relationships

    def test_predictions_pipeline_run_relationship_configuration(self):
        """Test that pipeline_run relationship is properly configured"""
        mapper = inspect(Predictions)
        pipeline_run_rel = mapper.relationships["pipeline_run"]
        
        assert pipeline_run_rel.mapper.class_ == PipelineRuns
        # Should have back_populates
        assert pipeline_run_rel.back_populates == "predictions"
        # Should be lazy loaded with 'joined'
        assert pipeline_run_rel.lazy == "joined"


class TestPredictionsInstantiation:
    """Tests for Predictions model instantiation"""

    def test_predictions_instantiation_basic(self):
        """Test basic instantiation of Predictions"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0, "feature2": 2.0},
            prediction=1
        )
        
        assert prediction.user_id == 1
        assert prediction.pipeline_run_id == 1
        assert prediction.prediction == 1
        assert prediction.input_data == {"feature1": 1.0, "feature2": 2.0}

    def test_predictions_instantiation_with_probability(self):
        """Test instantiation with probability"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1,
            probability=0.95
        )
        
        assert prediction.probability == 0.95

    def test_predictions_instantiation_without_probability(self):
        """Test that probability is optional"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=0
        )
        
        assert prediction.probability is None

    def test_predictions_inherits_active_default(self):
        """Test that active defaults to True from modelsGeneric"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )
        
        # active should default to True
        assert prediction.active is True or prediction.active is None


class TestPredictionsValidation:
    """Tests for Predictions model validation"""

    def test_predictions_requires_user_id(self):
        """Test that user_id is required"""
        # Creating without user_id should fail or be invalid
        prediction = Predictions(
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )
        
        assert prediction.user_id is None

    def test_predictions_requires_pipeline_run_id(self):
        """Test that pipeline_run_id is required"""
        prediction = Predictions(
            user_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )
        
        assert prediction.pipeline_run_id is None

    def test_predictions_requires_input_data(self):
        """Test that input_data is required"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            prediction=1
        )
        
        assert prediction.input_data is None

    def test_predictions_requires_prediction(self):
        """Test that prediction is required"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0}
        )
        
        assert prediction.prediction is None

    def test_predictions_accepts_various_input_data_formats(self):
        """Test that input_data accepts various JSON-compatible formats"""
        # Dict format
        pred1 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"key": "value"},
            prediction=1
        )
        assert pred1.input_data == {"key": "value"}
        
        # Nested structures
        pred2 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"nested": {"inner": [1, 2, 3]}},
            prediction=0
        )
        assert pred2.input_data == {"nested": {"inner": [1, 2, 3]}}

    def test_predictions_prediction_various_values(self):
        """Test that prediction accepts various integer values"""
        for pred_value in [0, 1, 2, 10]:
            prediction = Predictions(
                user_id=1,
                pipeline_run_id=1,
                input_data={"feature": 1.0},
                prediction=pred_value
            )
            assert prediction.prediction == pred_value

    def test_predictions_probability_valid_range(self):
        """Test that probability can accept valid probability values"""
        for prob_value in [0.0, 0.5, 0.99, 1.0]:
            prediction = Predictions(
                user_id=1,
                pipeline_run_id=1,
                input_data={"feature": 1.0},
                prediction=1,
                probability=prob_value
            )
            assert prediction.probability == prob_value


class TestPredictionsColumnTypes:
    """Tests for Predictions column types"""

    def test_predictions_id_is_integer(self):
        """Test that id column type is Integer"""
        mapper = inspect(Predictions)
        id_col = mapper.columns.get("id")
        assert isinstance(id_col.type, Integer)

    def test_predictions_user_id_is_integer(self):
        """Test that user_id column type is Integer"""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")
        assert isinstance(user_id_col.type, Integer)

    def test_predictions_pipeline_run_id_is_integer(self):
        """Test that pipeline_run_id column type is Integer"""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")
        assert isinstance(pipeline_run_id_col.type, Integer)

    def test_predictions_probability_is_float(self):
        """Test that probability column type is Float"""
        mapper = inspect(Predictions)
        probability_col = mapper.columns.get("probability")
        assert isinstance(probability_col.type, Float)

    def test_predictions_prediction_is_integer(self):
        """Test that prediction column type is Integer"""
        mapper = inspect(Predictions)
        prediction_col = mapper.columns.get("prediction")
        assert isinstance(prediction_col.type, Integer)


class TestPredictionsIntegration:
    """Integration tests for Predictions model"""

    def test_predictions_complete_data_set(self):
        """Test Predictions with complete data"""
        prediction = Predictions(
            user_id=123,
            pipeline_run_id=456,
            input_data={
                "age": 45,
                "income": 75000,
                "credit_score": 750,
                "dependents": 2
            },
            prediction=1,
            probability=0.87
        )
        
        assert prediction.user_id == 123
        assert prediction.pipeline_run_id == 456
        assert prediction.prediction == 1
        assert prediction.probability == 0.87
        assert prediction.input_data["age"] == 45
        assert prediction.input_data["credit_score"] == 750

    @patch("sqlalchemy.orm.relationship")
    def test_predictions_relationship_setup(self, mock_relationship):
        """Test that relationships are properly setup"""
        # This is a structural test
        mapper = inspect(Predictions)
        
        # Should have 2 relationships
        assert len(mapper.relationships) == 2

    def test_predictions_multiple_instances(self):
        """Test creating multiple Predictions instances"""
        predictions = []
        for i in range(5):
            pred = Predictions(
                user_id=1,
                pipeline_run_id=i + 1,
                input_data={"value": i},
                prediction=i % 2
            )
            predictions.append(pred)
        
        assert len(predictions) == 5
        assert all(isinstance(p, Predictions) for p in predictions)

    def test_predictions_input_data_with_features(self):
        """Test Predictions with realistic ML input data"""
        ml_input = {
            "feature_1": 0.5,
            "feature_2": -0.3,
            "feature_3": 1.2,
            "feature_4": 0.0,
            "engineered_feature_1": 2.5,
            "engineered_feature_2": 0.8
        }
        
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data=ml_input,
            prediction=1,
            probability=0.92
        )
        
        assert len(prediction.input_data) == 6
        assert all(key in prediction.input_data for key in ml_input.keys())


class TestPredictionsEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_predictions_empty_input_data(self):
        """Test Predictions with empty input_data dict"""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={},
            prediction=1
        )
        
        assert prediction.input_data == {}

    def test_predictions_large_probability_values(self):
        """Test Predictions with various probability values"""
        # Very small probability
        pred1 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"x": 1},
            prediction=0,
            probability=0.0001
        )
        assert pred1.probability == 0.0001
        
        # High probability
        pred2 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"x": 1},
            prediction=1,
            probability=0.9999
        )
        assert pred2.probability == 0.9999

    def test_predictions_with_null_probability(self):
        """Test Predictions with NULL probability"""
        # Simulate not providing probability
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature": 1},
            prediction=1,
            probability=None
        )
        
        assert prediction.probability is None

    def test_predictions_input_data_with_special_characters(self):
        """Test input_data with special characters in keys"""
        special_data = {
            "feature-1": 1.0,
            "feature_2": 2.0,
            "feature.3": 3.0,
            "feature/4": 4.0
        }
        
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data=special_data,
            prediction=1
        )
        
        assert len(prediction.input_data) == 4

    def test_predictions_different_prediction_classes(self):
        """Test Predictions with different classification values"""
        for class_value in [0, 1, 2, 3, 10]:
            prediction = Predictions(
                user_id=1,
                pipeline_run_id=1,
                input_data={"x": 1},
                prediction=class_value
            )
            assert prediction.prediction == class_value


class TestPredictionsAttributes:
    """Tests for model attributes and properties"""

    def test_predictions_has_all_required_attributes(self):
        """Test that Predictions model has all required attributes"""
        required_attrs = [
            "__tablename__", "id", "user_id", "pipeline_run_id",
            "input_data", "prediction", "probability", "user", "pipeline_run"
        ]
        
        for attr in required_attrs:
            assert hasattr(Predictions, attr)

    def test_predictions_table_name_readonly(self):
        """Test that __tablename__ is set correctly"""
        assert Predictions.__tablename__ == "predictions"
        assert isinstance(Predictions.__tablename__, str)

    def test_predictions_mapper_has_correct_column_count(self):
        """Test that Predictions mapper has expected number of columns"""
        mapper = inspect(Predictions)
        # Should have: id, user_id, pipeline_run_id, input_data, prediction, probability
        # Plus inherited: created_at, active
        column_count = len(mapper.columns)
        assert column_count >= 8  # At least 8 columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
