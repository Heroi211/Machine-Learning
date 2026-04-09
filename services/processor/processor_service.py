import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.custom_logger import setup_pipeline_run_logging
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from services.utils import utcnow

logger = logging.getLogger(__name__)


def _sklearn_feature_names(model) -> list[str] | None:
    """Nomes de features esperados pelo estimador sklearn (Pipeline ou estimador único)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def _align_dataframe_to_model(model, df: pd.DataFrame) -> pd.DataFrame:
    names = _sklearn_feature_names(model)
    if not names:
        return df
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas em falta para predição (alinhar ao treino): " + ", ".join(missing)
        )
    return df[names]


def _baseline_clean_encode_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Espelha `Baseline.clean_and_encode` para uma linha (predição). Exige coluna `target` (dummy)."""
    df_clean = df.copy()
    if "dataset" in df_clean.columns:
        df_clean = df_clean.drop(columns=["dataset"])
    if "target" not in df_clean.columns:
        df_clean["target"] = 0

    non_numeric = df_clean.drop(columns="target").select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = df_clean.drop(columns="target").select_dtypes(include=[np.number]).columns.tolist()

    for col in non_numeric:
        if df_clean[col].isnull().any():
            mode = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode)

    for col in num_cols:
        if df_clean[col].isnull().any():
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median)

    if df_clean.isnull().sum().sum() > 0:
        raise ValueError("Valores nulos após imputação — complete o payload de features.")

    cat_cols = []
    for col in non_numeric:
        unique_vals = set(df_clean[col].unique())
        if unique_vals <= {True, False}:
            df_clean[col] = df_clean[col].astype(bool)
        else:
            cat_cols.append(col)

    if cat_cols:
        df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    return df_clean.drop(columns=["target"])


def _prepare_prediction_features(run: PipelineRuns, domain: str, features: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame de entrada como no treino:
    - feature_engineering: chaves minúsculas (como `FeatureEngineering.load_data`) + strategy.build
    - baseline: chaves como enviadas (mesmos nomes que no CSV de treino) + clean/encode
    """
    ptype = run.pipeline_type
    d = domain.strip().lower()

    if ptype == "feature_engineering":
        from services.pipelines.feature_strategies import STRATEGY_REGISTRY

        if d not in STRATEGY_REGISTRY:
            raise ValueError(f"Domínio {domain!r} sem strategy. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")
        row = {str(k).strip().lower(): v for k, v in features.items()}
        row.pop("target", None)
        df = pd.DataFrame([row])
        strategy = STRATEGY_REGISTRY[d]()
        strategy.validate(df)
        return strategy.build(df)

    if ptype == "baseline":
        row = {str(k).strip(): v for k, v in features.items()}
        row.pop("target", None)
        df = pd.DataFrame([row])
        return _baseline_clean_encode_predict(df)

    raise ValueError(f"pipeline_type não suportado para predição: {ptype!r}")


async def run_baseline(file: UploadFile, objective: str, user_id: int, db: AsyncSession) -> PipelineRuns:
    """Salva o CSV enviado, executa o Baseline e persiste o resultado."""
    from services.pipelines.baseline import Baseline
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(settings.path_data, exist_ok=True)
    input_path = os.path.join(settings.path_data, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(user_id=user_id, pipeline_type="baseline", objective=objective, status="processing", original_filename=file.filename)

    async with db as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

        try:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(settings.path_data, settings.path_logs, run_ts)
            setup_pipeline_run_logging(
                snapshot_path,
                run_ts,
                run_id=run.id,
                objective=objective,
                pipeline_type="baseline",
            )
            from services.pipelines.feature_strategies import get_class_labels
            pipeline = Baseline(
                pobjective=objective,
                run_timestamp=run_ts,
                csv_path=input_path,
                class_labels=get_class_labels(objective),
            )
            pipeline.run(start_time=datetime.now())
            pipeline.save_artifacts()

            model_path = os.path.join(settings.path_model, f"baseline_model_{objective}_{pipeline.now}.joblib")
            csv_path = os.path.join(settings.path_data_preprocessed, f"{objective}_sample_{pipeline.now}.csv")

            model = pipeline.model
            y_pred_test = model.predict(pipeline.x_test)

            metrics = {
                "test_accuracy": float(accuracy_score(pipeline.y_test, y_pred_test)),
                "test_f1": float(f1_score(pipeline.y_test, y_pred_test)),
                "test_precision": float(precision_score(pipeline.y_test, y_pred_test)),
                "test_recall": float(recall_score(pipeline.y_test, y_pred_test)),
            }

            run.status = "completed"
            run.metrics = metrics
            run.model_path = model_path
            run.csv_output_path = csv_path
            run.completed_at = utcnow()

        except Exception as e:
            logger.error(f"Baseline falhou: {e}")
            run.status = "failed"
            run.error_message = str(e)[:1000]
            run.completed_at = utcnow()
            run.active = False

        session.add(run)
        await session.commit()
        await session.refresh(run)

    return run


async def run_feature_engineering(file: UploadFile, objective: str, user_id: int, db: AsyncSession, optimization_metric: str = "accuracy", time_limit_minutes: int = 2, acc_target: float = 0.90) -> PipelineRuns:
    """Salva o CSV pré-processado, executa o Feature Engineering e persiste o resultado."""
    from services.pipelines.feature_engineering import FeatureEngineering
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY
    from services.pipelines.fe_model_selection import normalize_optimization_metric

    metric = normalize_optimization_metric(optimization_metric)

    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    input_path = os.path.join(settings.path_data_preprocessed, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(user_id=user_id, pipeline_type="feature_engineering", objective=objective, status="processing", original_filename=file.filename)

    async with db as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

        try:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(settings.path_data, settings.path_logs, run_ts)
            setup_pipeline_run_logging(snapshot_path, run_ts, run_id=run.id, objective=objective, pipeline_type="feature_engineering")
            strategy = STRATEGY_REGISTRY[objective]()
            pipeline = FeatureEngineering(objective=objective, strategy=strategy, run_timestamp=run_ts, csv_path=input_path, optimization_metric=metric)
            pipeline.run(time_limit_minutes=time_limit_minutes, acc_target=acc_target)

            run.status = "completed"
            merged_metrics = dict(pipeline.tuned_metrics)
            merged_metrics["optimization_metric"] = metric
            run.metrics = merged_metrics
            run.model_path = os.path.join(settings.path_model, f"best_{objective}_{pipeline.now}.joblib")
            run.csv_output_path = os.path.abspath(input_path)
            run.completed_at = utcnow()

        except Exception as e:
            logger.error(f"Feature Engineering falhou: {e}")
            run.status = "failed"
            run.error_message = str(e)[:1000]
            run.completed_at = utcnow()
            run.active = False

        session.add(run)
        await session.commit()
        await session.refresh(run)

    return run


async def predict_for_domain(domain: str, features: dict, user_id: int, db: AsyncSession) -> Predictions:
    """Resolve o modelo ativo do domínio e prediz para um indivíduo."""
    from services.processor.deployment_service import NoActiveDeploymentError, get_active_deployment

    async with db as session:
        deployment = await get_active_deployment(domain, session)
        if not deployment:
            raise NoActiveDeploymentError(
                f"Nenhum modelo ativo para o domínio '{domain}'. Promova um pipeline concluído (admin)."
            )

        run = deployment.pipeline_run
        if not run.model_path or not os.path.exists(run.model_path):
            raise ValueError(f"Modelo não encontrado em: {run.model_path}")

        model = joblib.load(run.model_path)

        df_input = _prepare_prediction_features(run, domain, features)
        df_input = _align_dataframe_to_model(model, df_input)

        prediction_value = int(model.predict(df_input)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[0]
            probability = float(proba[1])

        pred = Predictions(
            user_id=user_id,
            pipeline_run_id=run.id,
            input_data=features,
            prediction=prediction_value,
            probability=probability,
        )

        session.add(pred)
        await session.commit()
        await session.refresh(pred)

    return pred
