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
    """
    Alinha uma linha ao mesmo esquema de features do treino do Baseline.
    Imputação e one-hot ficam no ``Pipeline`` persistido (``model.predict``).
    """
    df_clean = df.copy()
    if "dataset" in df_clean.columns:
        df_clean = df_clean.drop(columns=["dataset"])
    if "target" in df_clean.columns:
        df_clean = df_clean.drop(columns=["target"])
    for col in list(df_clean.columns):
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(np.int8)
    return df_clean


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
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

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
            y_proba_test = model.predict_proba(pipeline.x_test)[:, 1]
            _zd = {"zero_division": 0}
            yt = pipeline.y_test
            metrics = {
                "test_accuracy": float(accuracy_score(yt, y_pred_test)),
                "test_f1": float(f1_score(yt, y_pred_test, **_zd)),
                "test_precision": float(precision_score(yt, y_pred_test, **_zd)),
                "test_recall": float(recall_score(yt, y_pred_test, **_zd)),
            }
            if int(yt.sum()) > 0 and int(len(yt) - yt.sum()) > 0:
                metrics["test_pr_auc"] = float(average_precision_score(yt, y_proba_test))

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


async def run_feature_engineering(
    file: UploadFile,
    objective: str,
    user_id: int,
    db: AsyncSession,
    optimization_metric: str = "accuracy",
    time_limit_minutes: int = 2,
    acc_target: float = 0.90,
) -> tuple[PipelineRuns, str | None]:
    """
    Baseline (mesmo ficheiro de entrada) + FE; gera um ZIP de artefatos.
    Retorna (PipelineRuns, path_do_zip) — ``path_do_zip`` é None se falhar antes do zip.

    Tuning: em rotas síncronas aplica teto ``settings.sync_fe_tune_max_minutes`` (não Airflow).
    """
    import glob
    import shutil
    import tempfile
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    from services.pipelines.baseline import Baseline
    from services.pipelines.feature_engineering import FeatureEngineering
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY, get_class_labels
    from services.pipelines.fe_model_selection import normalize_optimization_metric
    from services.processor.artifact_bundle import (
        build_manifest,
        copy_if_exists,
        glob_copy_pattern,
        safe_rmtree,
        safe_unlink,
        write_manifest,
        zip_tree,
    )
    from services.processor.deployment_service import get_active_deployment

    metric = normalize_optimization_metric(optimization_metric)

    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    os.makedirs(settings.path_data, exist_ok=True)
    os.makedirs(settings.path_data_preprocessed, exist_ok=True)

    content = await file.read()
    input_path = os.path.join(settings.path_data, f"fe_{user_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(content)

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="feature_engineering",
        objective=objective,
        status="processing",
        original_filename=file.filename,
    )
    zip_path: str | None = None
    run_root: str | None = None

    async with db as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)

    effective_tuning_minutes = min(time_limit_minutes, settings.sync_fe_tune_max_minutes)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(settings.path_data, settings.path_logs, f"{run_ts}_fe{run.id}")

    try:
        setup_pipeline_run_logging(
            snapshot_path, run_ts, run_id=run.id, objective=objective, pipeline_type="feature_engineering"
        )

        run_root = tempfile.mkdtemp(prefix=f"fe_bundle_{run.id}_")
        b_in = os.path.join(run_root, "00_input_baseline")
        b_out = os.path.join(run_root, "10_baseline")
        fe_d = os.path.join(run_root, "20_feature_engineering")
        os.makedirs(b_in, exist_ok=True)
        shutil.copy2(input_path, os.path.join(b_in, file.filename))

        # --- Baseline (sem save_artifacts: evita apagar o CSV cedo) ---
        pipeline_b = Baseline(
            pobjective=objective,
            run_timestamp=run_ts,
            csv_path=input_path,
            class_labels=get_class_labels(objective),
        )
        pipeline_b.run(start_time=datetime.now())

        model_baseline = os.path.join(settings.path_model, f"baseline_model_{objective}_{pipeline_b.now}.joblib")
        csv_baseline = os.path.join(
            settings.path_data_preprocessed, f"{objective}_sample_{pipeline_b.now}.csv"
        )
        _zd = {"zero_division": 0}
        yt = pipeline_b.y_test
        y_pred_bt = pipeline_b.model.predict(pipeline_b.x_test)
        y_proba = pipeline_b.model.predict_proba(pipeline_b.x_test)[:, 1]
        baseline_metrics = {
            "test_accuracy": float(accuracy_score(yt, y_pred_bt)),
            "test_f1": float(f1_score(yt, y_pred_bt, **_zd)),
            "test_precision": float(precision_score(yt, y_pred_bt, **_zd)),
            "test_recall": float(recall_score(yt, y_pred_bt, **_zd)),
        }
        if int(yt.sum()) > 0 and int(len(yt) - yt.sum()) > 0:
            baseline_metrics["test_pr_auc"] = float(
                average_precision_score(yt, y_proba)
            )

        copy_if_exists(model_baseline, b_out)
        copy_if_exists(csv_baseline, b_out)
        glob_copy_pattern(
            settings.path_graphs, f"*{pipeline_b.now}*.png", os.path.join(b_out, "graphs")
        )
        for extra in (
            f"outliers_boxplot_{pipeline_b.now}.png",
            f"target_distribution_pie_{pipeline_b.now}.png",
            f"target_distribution_bar_{pipeline_b.now}.png",
        ):
            p = os.path.join(settings.path_graphs, extra)
            copy_if_exists(p, os.path.join(b_out, "graphs"))
        copy_if_exists(
            os.path.join(settings.path_graphs, f"data_view_{pipeline_b.now}.csv"),
            b_out,
        )

        if not os.path.isfile(csv_baseline):
            raise FileNotFoundError(
                f"CSV pré-processado do Baseline inesperadamente inexistente: {csv_baseline}"
            )

        # --- Feature Engineering (entrada = saída do Baseline) ---
        fe_plots = os.path.join(fe_d, "plots")
        os.makedirs(fe_d, exist_ok=True)
        strategy = STRATEGY_REGISTRY[objective]()
        pipeline = FeatureEngineering(
            objective=objective,
            strategy=strategy,
            run_timestamp=run_ts,
            csv_path=csv_baseline,
            optimization_metric=metric,
            export_figures_dir=fe_plots,
        )
        pipeline.run(time_limit_minutes=effective_tuning_minutes, acc_target=acc_target)

        fe_joblib = os.path.join(settings.path_model, f"best_{objective}_{pipeline.now}.joblib")
        copy_if_exists(fe_joblib, fe_d)
        with open(
            os.path.join(fe_d, "best_model_name.txt"), "w", encoding="utf-8"
        ) as tf:
            tf.write(pipeline.best_model_name or "")

        # --- manifest + zip ---
        active_dep = None
        async with db as s2:
            ad = await get_active_deployment(objective, s2)
            if ad is not None:
                active_dep = ad.id

        merged_metrics: dict = {
            "baseline": baseline_metrics,
            "optimization_metric": metric,
            "tuning_time_limit_effective_minutes": effective_tuning_minutes,
            "tuning_time_limit_requested": time_limit_minutes,
        }
        merged_metrics.update(dict(pipeline.tuned_metrics))
        if pipeline.best_model_name:
            merged_metrics["best_model_name"] = pipeline.best_model_name

        all_files: list[str] = []
        for root, _dirs, files in os.walk(run_root):
            for name in files:
                all_files.append(os.path.join(root, name))

        manifest_path = os.path.join(run_root, "manifest.json")
        man = build_manifest(
            pipeline_run_id=run.id,
            objective=objective,
            run_timestamp=run_ts,
            status="completed",
            input_filename=file.filename,
            input_path=input_path,
            paths_in_bundle=all_files,
            metrics=merged_metrics,
            mlflow_baseline_run_id=pipeline_b.mlflow_run_id,
            mlflow_fe_run_id=pipeline.mlflow_run_id,
            best_model_name=pipeline.best_model_name,
            original_filename=file.filename,
            active_deployment_id=active_dep,
        )
        write_manifest(man, manifest_path)
        man = build_manifest(
            pipeline_run_id=run.id,
            objective=objective,
            run_timestamp=run_ts,
            status="completed",
            input_filename=file.filename,
            input_path=input_path,
            paths_in_bundle=all_files + [manifest_path],
            metrics=merged_metrics,
            mlflow_baseline_run_id=pipeline_b.mlflow_run_id,
            mlflow_fe_run_id=pipeline.mlflow_run_id,
            best_model_name=pipeline.best_model_name,
            original_filename=file.filename,
            active_deployment_id=active_dep,
        )
        write_manifest(man, manifest_path)

        zip_path = os.path.join(tempfile.gettempdir(), f"fe_artifacts_{run.id}_{run_ts}.zip")
        if os.path.isfile(zip_path):
            safe_unlink(zip_path)
        zip_tree(run_root, zip_path)

        run.status = "completed"
        run.model_path = fe_joblib
        run.csv_output_path = None
        run.metrics = merged_metrics
        run.completed_at = utcnow()

        # --- limpeza: não apagar ``fe_joblib`` (referência do run / predição); não tocar MLflow ---
        safe_unlink(model_baseline)
        for g in glob.glob(os.path.join(settings.path_graphs, f"*{pipeline_b.now}*")):
            safe_unlink(g)
        safe_unlink(csv_baseline)

    except Exception as e:
        logger.error(f"Feature Engineering falhou: {e}", exc_info=True)
        run.status = "failed"
        run.error_message = str(e)[:1000]
        run.completed_at = utcnow()
        run.active = False
    finally:
        if run_root:
            safe_rmtree(run_root)
        if input_path and os.path.isfile(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

    async with db as session:
        merged = await session.merge(run)
        await session.commit()
        await session.refresh(merged)

    return merged, zip_path


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
