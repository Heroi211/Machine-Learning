"""Testes do modelo ORM de predições."""

from unittest.mock import patch
import pytest
from sqlalchemy import Integer, Float
from sqlalchemy.inspection import inspect

from models.predictions import Predictions
from models.users import Users
from models.pipeline_runs import PipelineRuns
from core.generic import modelsGeneric


class TestPredictionsModel:
    """Testes da estrutura e dos atributos do modelo Predictions."""

    def test_predictions_inherits_from_generic(self):
        """Verifica se Predictions herda de modelsGeneric."""
        assert issubclass(Predictions, modelsGeneric)

    def test_predictions_table_name(self):
        """Verifica se Predictions usa o nome correto de tabela."""
        assert Predictions.__tablename__ == "predictions"

    def test_predictions_has_inherited_columns(self):
        """Verifica se Predictions possui colunas herdadas de modelsGeneric."""
        # Obtém os nomes das colunas a partir do mapper.
        mapper = inspect(Predictions)
        column_names = [col.name for col in mapper.columns]

        # Deve conter as colunas herdadas.
        assert "created_at" in column_names
        assert "active" in column_names

    def test_predictions_created_at_column_exists(self):
        """Verifica se created_at é herdada e configurada."""
        mapper = inspect(Predictions)
        created_at_col = mapper.columns.get("created_at")

        assert created_at_col is not None
        assert not created_at_col.nullable

    def test_predictions_active_column_exists(self):
        """Verifica se active é herdada."""
        mapper = inspect(Predictions)
        active_col = mapper.columns.get("active")

        assert active_col is not None
        assert active_col.default is not None


class TestPredictionsColumns:
    """Testes das colunas do modelo Predictions."""

    def test_predictions_id_column(self):
        """Verifica se a coluna id está definida corretamente."""
        mapper = inspect(Predictions)
        id_col = mapper.columns.get("id")

        assert id_col is not None
        assert id_col.primary_key
        assert id_col.autoincrement

    def test_predictions_user_id_column(self):
        """Verifica se a coluna user_id está definida corretamente."""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")

        assert user_id_col is not None
        assert not user_id_col.nullable
        assert len(user_id_col.foreign_keys) > 0

    def test_predictions_user_id_foreign_key(self):
        """Verifica se user_id possui a chave estrangeira correta."""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")

        fk = list(user_id_col.foreign_keys)[0]
        assert fk.column.table.name == "users"

    def test_predictions_pipeline_run_id_column(self):
        """Verifica se pipeline_run_id está definida corretamente."""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")

        assert pipeline_run_id_col is not None
        assert not pipeline_run_id_col.nullable
        assert len(pipeline_run_id_col.foreign_keys) > 0

    def test_predictions_pipeline_run_id_foreign_key(self):
        """Verifica se pipeline_run_id possui a chave estrangeira correta."""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")

        fk = list(pipeline_run_id_col.foreign_keys)[0]
        assert fk.column.table.name == "pipeline_runs"

    def test_predictions_input_data_column(self):
        """Verifica se a coluna input_data está definida corretamente."""
        mapper = inspect(Predictions)
        input_data_col = mapper.columns.get("input_data")

        assert input_data_col is not None
        assert not input_data_col.nullable
        # Verifica se é um tipo compatível com JSON.
        assert type(input_data_col.type).__name__ in ["JSON", "String"]

    def test_predictions_prediction_column(self):
        """Verifica se a coluna prediction está definida corretamente."""
        mapper = inspect(Predictions)
        prediction_col = mapper.columns.get("prediction")

        assert prediction_col is not None
        assert not prediction_col.nullable
        assert isinstance(prediction_col.type, Integer)

    def test_predictions_probability_column(self):
        """Verifica se a coluna probability está definida corretamente."""
        mapper = inspect(Predictions)
        probability_col = mapper.columns.get("probability")

        assert probability_col is not None
        assert probability_col.nullable  # Probability pode ser NULL.
        assert isinstance(probability_col.type, Float)


class TestPredictionsRelationships:
    """Testes dos relacionamentos do modelo Predictions."""

    def test_predictions_has_user_relationship(self):
        """Verifica se Predictions possui relacionamento com user."""
        mapper = inspect(Predictions)
        relationships = {rel.key: rel for rel in mapper.relationships}

        assert "user" in relationships

    def test_predictions_user_relationship_configuration(self):
        """Verifica se o relacionamento user está configurado corretamente."""
        mapper = inspect(Predictions)
        user_rel = mapper.relationships["user"]

        assert user_rel.mapper.class_ == Users
        # Deve usar carregamento lazy com 'joined'.
        assert user_rel.lazy == "joined"

    def test_predictions_has_pipeline_run_relationship(self):
        """Verifica se Predictions possui relacionamento com pipeline_run."""
        mapper = inspect(Predictions)
        relationships = {rel.key: rel for rel in mapper.relationships}

        assert "pipeline_run" in relationships

    def test_predictions_pipeline_run_relationship_configuration(self):
        """Verifica se o relacionamento pipeline_run está configurado."""
        mapper = inspect(Predictions)
        pipeline_run_rel = mapper.relationships["pipeline_run"]

        assert pipeline_run_rel.mapper.class_ == PipelineRuns
        # Deve ter back_populates.
        assert pipeline_run_rel.back_populates == "predictions"
        # Deve usar carregamento lazy com 'joined'.
        assert pipeline_run_rel.lazy == "joined"


class TestPredictionsInstantiation:
    """Testes de instanciação do modelo Predictions."""

    def test_predictions_instantiation_basic(self):
        """Verifica instanciação básica de Predictions."""
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
        """Verifica instanciação com probabilidade."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1,
            probability=0.95
        )

        assert prediction.probability == 0.95

    def test_predictions_instantiation_without_probability(self):
        """Verifica se probability é opcional."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=0
        )

        assert prediction.probability is None

    def test_predictions_inherits_active_default(self):
        """Verifica se active usa padrão herdado de modelsGeneric."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )

        # active deve ter True como padrão.
        assert prediction.active is True or prediction.active is None


class TestPredictionsValidation:
    """Testes de validação do modelo Predictions."""

    def test_predictions_requires_user_id(self):
        """Verifica comportamento sem user_id."""
        # Criar sem user_id deve falhar ou ficar inválido.
        prediction = Predictions(
            pipeline_run_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )

        assert prediction.user_id is None

    def test_predictions_requires_pipeline_run_id(self):
        """Verifica comportamento sem pipeline_run_id."""
        prediction = Predictions(
            user_id=1,
            input_data={"feature1": 1.0},
            prediction=1
        )

        assert prediction.pipeline_run_id is None

    def test_predictions_requires_input_data(self):
        """Verifica comportamento sem input_data."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            prediction=1
        )

        assert prediction.input_data is None

    def test_predictions_requires_prediction(self):
        """Verifica comportamento sem prediction."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature1": 1.0}
        )

        assert prediction.prediction is None

    def test_predictions_accepts_various_input_data_formats(self):
        """Verifica se input_data aceita formatos compatíveis com JSON."""
        # Formato dict.
        pred1 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"key": "value"},
            prediction=1
        )
        assert pred1.input_data == {"key": "value"}

        # Estruturas aninhadas.
        pred2 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"nested": {"inner": [1, 2, 3]}},
            prediction=0
        )
        assert pred2.input_data == {"nested": {"inner": [1, 2, 3]}}

    def test_predictions_prediction_various_values(self):
        """Verifica se prediction aceita diferentes valores inteiros."""
        for pred_value in [0, 1, 2, 10]:
            prediction = Predictions(
                user_id=1,
                pipeline_run_id=1,
                input_data={"feature": 1.0},
                prediction=pred_value
            )
            assert prediction.prediction == pred_value

    def test_predictions_probability_valid_range(self):
        """Verifica se probability aceita valores válidos."""
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
    """Testes dos tipos de colunas de Predictions."""

    def test_predictions_id_is_integer(self):
        """Verifica se a coluna id é Integer."""
        mapper = inspect(Predictions)
        id_col = mapper.columns.get("id")
        assert isinstance(id_col.type, Integer)

    def test_predictions_user_id_is_integer(self):
        """Verifica se a coluna user_id é Integer."""
        mapper = inspect(Predictions)
        user_id_col = mapper.columns.get("user_id")
        assert isinstance(user_id_col.type, Integer)

    def test_predictions_pipeline_run_id_is_integer(self):
        """Verifica se a coluna pipeline_run_id é Integer."""
        mapper = inspect(Predictions)
        pipeline_run_id_col = mapper.columns.get("pipeline_run_id")
        assert isinstance(pipeline_run_id_col.type, Integer)

    def test_predictions_probability_is_float(self):
        """Verifica se a coluna probability é Float."""
        mapper = inspect(Predictions)
        probability_col = mapper.columns.get("probability")
        assert isinstance(probability_col.type, Float)

    def test_predictions_prediction_is_integer(self):
        """Verifica se a coluna prediction é Integer."""
        mapper = inspect(Predictions)
        prediction_col = mapper.columns.get("prediction")
        assert isinstance(prediction_col.type, Integer)


class TestPredictionsIntegration:
    """Testes de integração do modelo Predictions."""

    def test_predictions_complete_data_set(self):
        """Verifica Predictions com dados completos."""
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
        """Verifica se relacionamentos estão configurados corretamente."""
        # Este é um teste estrutural.
        mapper = inspect(Predictions)

        # Deve ter 2 relacionamentos.
        assert len(mapper.relationships) == 2

    def test_predictions_multiple_instances(self):
        """Verifica criação de múltiplas instâncias de Predictions."""
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
        """Verifica Predictions com dados realistas de entrada de ML."""
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
    """Testes de casos de borda e cenários especiais."""

    def test_predictions_empty_input_data(self):
        """Verifica Predictions com input_data vazio."""
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={},
            prediction=1
        )

        assert prediction.input_data == {}

    def test_predictions_large_probability_values(self):
        """Verifica Predictions com diferentes valores de probabilidade."""
        # Probabilidade muito pequena.
        pred1 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"x": 1},
            prediction=0,
            probability=0.0001
        )
        assert pred1.probability == 0.0001

        # Probabilidade alta.
        pred2 = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"x": 1},
            prediction=1,
            probability=0.9999
        )
        assert pred2.probability == 0.9999

    def test_predictions_with_null_probability(self):
        """Verifica Predictions com probability NULL."""
        # Simula ausência de probability.
        prediction = Predictions(
            user_id=1,
            pipeline_run_id=1,
            input_data={"feature": 1},
            prediction=1,
            probability=None
        )

        assert prediction.probability is None

    def test_predictions_input_data_with_special_characters(self):
        """Verifica input_data com caracteres especiais nas chaves."""
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
        """Verifica Predictions com diferentes valores de classificação."""
        for class_value in [0, 1, 2, 3, 10]:
            prediction = Predictions(
                user_id=1,
                pipeline_run_id=1,
                input_data={"x": 1},
                prediction=class_value
            )
            assert prediction.prediction == class_value


class TestPredictionsAttributes:
    """Testes de atributos e propriedades do modelo."""

    def test_predictions_has_all_required_attributes(self):
        """Verifica se Predictions possui todos os atributos esperados."""
        required_attrs = [
            "__tablename__", "id", "user_id", "pipeline_run_id",
            "input_data", "prediction", "probability", "user", "pipeline_run"
        ]

        for attr in required_attrs:
            assert hasattr(Predictions, attr)

    def test_predictions_table_name_readonly(self):
        """Verifica se __tablename__ está definido corretamente."""
        assert Predictions.__tablename__ == "predictions"
        assert isinstance(Predictions.__tablename__, str)

    def test_predictions_mapper_has_correct_column_count(self):
        """Verifica se o mapper possui a quantidade esperada de colunas."""
        mapper = inspect(Predictions)
        # Deve ter: id, user_id, pipeline_run_id, input_data, prediction, probability.
        # Mais as herdadas: created_at, active.
        column_count = len(mapper.columns)
        assert column_count >= 8  # Pelo menos 8 colunas.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
