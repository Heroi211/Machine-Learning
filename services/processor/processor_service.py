import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.custom_logger import setup_pipeline_run_logging
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from services.utils import utcnow

logger = logging.getLogger(__name__)


async def run_baseline(file: UploadFile,objective: str,user_id: int,db: AsyncSession) -> PipelineRuns:
    """Salva o CSV enviado, executa o Baseline e persiste o resultado."""
    from services.pipelines.baseline import Baseline
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(settings.path_data, exist_ok=True)
    input_path = os.path.join(settings.path_data, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(user_id=user_id,pipeline_type="baseline",objective=objective,status="processing",original_filename=file.filename)

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
            pipeline = Baseline(pobjective=objective, run_timestamp=run_ts, csv_path=input_path)
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

        session.add(run)
        await session.commit()
        await session.refresh(run)

    return run


async def run_feature_engineering(file: UploadFile,objective: str,user_id: int,db: AsyncSession) -> PipelineRuns:
    """Salva o CSV pré-processado, executa o Feature Engineering e persiste o resultado."""
    from services.pipelines.feature_engineering import FeatureEngineering
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY

    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    input_path = os.path.join(settings.path_data_preprocessed, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(user_id=user_id,pipeline_type="feature_engineering",objective=objective,status="processing",original_filename=file.filename)

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
                pipeline_type="feature_engineering",
            )
            strategy = STRATEGY_REGISTRY[objective]()
            pipeline = FeatureEngineering(
                objective=objective, strategy=strategy, run_timestamp=run_ts, csv_path=input_path
            )
            pipeline.run()

            run.status = "completed"
            run.metrics = pipeline.tuned_metrics
            run.model_path = os.path.join(settings.path_model, f"best_{objective}_{pipeline.now}.joblib")
            run.csv_output_path = os.path.abspath(input_path)
            run.completed_at = utcnow()

        except Exception as e:
            logger.error(f"Feature Engineering falhou: {e}")
            run.status = "failed"
            run.error_message = str(e)[:1000]
            run.completed_at = utcnow()

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

        df_input = pd.DataFrame([features])

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
