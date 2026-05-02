import logging
import os
import json
import re
import shutil
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


_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY = "strategy_monthly_charges_median"


def _hydrate_strategy_from_run_metrics(strategy, run: PipelineRuns) -> None:
    """Recarrega estado da strategy guardado em ``run.metrics`` (ex.: mediana no Churn)."""
    m = run.metrics
    if not m:
        return
    if _STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY in m and hasattr(strategy, "monthly_median"):
        try:
            strategy.monthly_median = float(m[_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY])
        except (TypeError, ValueError):
            pass


def _infer_train_matrix_payload(features: dict) -> None:
    """Levanta erro claro se o cliente enviar colunas típicas de ``train_model_input`` (pós-OHE / pós-escala)."""
    keys = [str(k).strip().lower() for k in features]
    ohe_like = (
        "internetservice_",
        "contract_",
        "paymentmethod_",
    )
    if any(any(x.startswith(p) for p in ohe_like) for x in keys):
        raise ValueError(
            "O corpo parece estar no formato da matriz **pós-transformação** (ex.: colunas tipo "
            "`internetservice_Fiber optic`, `contract_One year`). O endpoint /predict espera o nível **pré-ColumnTransformer**: "
            "o mesmo de `train_features_pre_transform.csv`, **sem** colunas derivadas da strategy "
            "(ex.: sem `is_new_customer`, `tenure_log`) e com valores **cruos** em `monthlycharges`/`totalcharges`, "
            "não normalizados. Ver documentação em `ChurnFeaturesInput`."
        )


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
        _infer_train_matrix_payload(row)
        df = pd.DataFrame([row])
        strategy = STRATEGY_REGISTRY[d]()
        _hydrate_strategy_from_run_metrics(strategy, run)
        strategy.validate(df)
        return strategy.build(df)

    if ptype == "baseline":
        row = {str(k).strip(): v for k, v in features.items()}
        row.pop("target", None)
        df = pd.DataFrame([row])
        return _baseline_clean_encode_predict(df)

    raise ValueError(f"pipeline_type não suportado para predição: {ptype!r}")


def _extract_timestamp_from_baseline_model_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    m = re.search(r"baseline_model_[^_]+_(\d{8}_\d{6})\.joblib$", os.path.basename(model_path))
    if m:
        return m.group(1)
    return None


async def _resolve_manifest_path_for_fe(
    db: AsyncSession,
    objective: str,
    manifest_path: str | None,
    baseline_run_id: int | None,
) -> str:
    if manifest_path and baseline_run_id:
        raise ValueError("Informe apenas um entre manifest_path e baseline_run_id.")

    if manifest_path:
        candidate = os.path.abspath(manifest_path)
        if not os.path.isfile(candidate):
            raise ValueError(f"Manifest informado não existe: {candidate}")
        return candidate

    if baseline_run_id is not None:
        async with db as session:
            run = await session.get(PipelineRuns, baseline_run_id)
        if not run:
            raise ValueError(f"baseline_run_id não encontrado: {baseline_run_id}")
        if run.pipeline_type != "baseline":
            raise ValueError(f"pipeline_run_id {baseline_run_id} não é baseline.")
        if run.objective != objective:
            raise ValueError(
                f"baseline_run_id {baseline_run_id} com objective='{run.objective}', esperado '{objective}'."
            )

        ts = _extract_timestamp_from_baseline_model_path(run.model_path)
        if not ts:
            raise ValueError(
                "Não foi possível inferir o timestamp do baseline pelo model_path. Informe manifest_path explicitamente."
            )
        candidate = os.path.join(settings.path_data, settings.path_logs, ts, "manifest.json")
        if not os.path.isfile(candidate):
            raise ValueError(
                f"Manifest do baseline_run_id {baseline_run_id} não encontrado em: {candidate}"
            )
        return candidate

    candidate = os.path.join(settings.path_data_preprocessed, "manifest.json")
    if not os.path.isfile(candidate):
        raise ValueError(f"Manifest do baseline não encontrado: {candidate}")
    return candidate


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
            csv_path = os.path.join(settings.path_data_preprocessed, pipeline.contract_sample_name)

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
    objective: str,
    user_id: int,
    db: AsyncSession,
    optimization_metric: str = "accuracy",
    min_precision: float | None = None,
    min_roc_auc: float | None = None,
    tuning_n_iter: int | None = None,
    manifest_path: str | None = None,
    baseline_run_id: int | None = None,
    time_limit_minutes: int = 2,
    acc_target: float | None = None,
) -> tuple[PipelineRuns, str | None]:
    """
    Executa apenas o pipeline de Feature Engineering usando o contrato
    produzido pelo baseline (manifest + sample), sem rerun do baseline.
    """
    import shutil
    import tempfile

    from services.pipelines.feature_engineering import FeatureEngineering
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY
    from services.pipelines.fe_model_selection import normalize_optimization_metric
    from services.processor.artifact_bundle import (
        build_manifest,
        copy_if_exists,
        safe_rmtree,
        safe_unlink,
        write_manifest,
        zip_tree,
    )
    from services.processor.deployment_service import get_active_deployment

    metric = normalize_optimization_metric(optimization_metric)
    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    resolved_manifest_path = await _resolve_manifest_path_for_fe(
        db=db,
        objective=objective,
        manifest_path=manifest_path,
        baseline_run_id=baseline_run_id,
    )
    with open(resolved_manifest_path, "r", encoding="utf-8") as f:
        baseline_manifest = json.load(f)

    manifest_objective = str(baseline_manifest.get("objective", "")).strip().lower()
    if manifest_objective != objective:
        raise ValueError(
            f"Manifest do baseline com objective='{manifest_objective}', mas FE foi solicitado para '{objective}'."
        )
    if baseline_manifest.get("sample_schema") != "raw_clean":
        raise ValueError(
            f"sample_schema inválido no manifest: {baseline_manifest.get('sample_schema')!r}. Esperado: 'raw_clean'."
        )
    csv_baseline = baseline_manifest.get("output_sample_csv_stable")
    if not csv_baseline:
        raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")
    csv_baseline = os.path.abspath(csv_baseline)
    if not os.path.isfile(csv_baseline):
        raise FileNotFoundError(f"CSV do baseline não encontrado: {csv_baseline}")

    baseline_input_path = baseline_manifest.get("input_csv_snapshot") or baseline_manifest.get("input_csv_source")
    baseline_input_path = os.path.abspath(baseline_input_path) if baseline_input_path else None
    original_filename = os.path.basename(baseline_input_path) if baseline_input_path else os.path.basename(csv_baseline)

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="feature_engineering",
        objective=objective,
        status="processing",
        original_filename=original_filename,
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
        os.makedirs(fe_d, exist_ok=True)

        if baseline_input_path and os.path.isfile(baseline_input_path):
            shutil.copy2(baseline_input_path, os.path.join(b_in, os.path.basename(baseline_input_path)))
        copy_if_exists(csv_baseline, b_out)
        copy_if_exists(resolved_manifest_path, b_out)
        model_baseline = baseline_manifest.get("model_path")
        if model_baseline:
            copy_if_exists(os.path.abspath(model_baseline), b_out)
        graphs_dir = baseline_manifest.get("graphs_dir")
        if graphs_dir and os.path.isdir(os.path.abspath(graphs_dir)):
            for name in os.listdir(os.path.abspath(graphs_dir)):
                copy_if_exists(os.path.join(os.path.abspath(graphs_dir), name), os.path.join(b_out, "graphs"))

        strategy = STRATEGY_REGISTRY[objective]()
        fe_plots = os.path.join(fe_d, "plots")
        pipeline = FeatureEngineering(
            objective=objective,
            strategy=strategy,
            run_timestamp=run_ts,
            csv_path=csv_baseline,
            manifest_path=resolved_manifest_path,
            optimization_metric=metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            export_figures_dir=fe_plots,
        )
        pipeline.run(time_limit_minutes=effective_tuning_minutes, acc_target=acc_target)

        fe_joblib = os.path.join(settings.path_model, f"best_{objective}_{pipeline.now}.joblib")
        copy_if_exists(fe_joblib, fe_d)
        # Conteúdo de _export_fe_bundle (CSVs + resumo PyTorch) — entra no ZIP em 20_feature_engineering/fe_export/
        fe_export_src = os.path.join(settings.path_model, f"fe_export_{objective}_{pipeline.now}")
        if os.path.isdir(fe_export_src):
            fe_export_dst = os.path.join(fe_d, "fe_export")
            shutil.copytree(fe_export_src, fe_export_dst, dirs_exist_ok=True)
        with open(os.path.join(fe_d, "best_model_name.txt"), "w", encoding="utf-8") as tf:
            tf.write(pipeline.best_model_name or "")

        active_dep = None
        async with db as s2:
            ad = await get_active_deployment(objective, s2)
            if ad is not None:
                active_dep = ad.id

        merged_metrics: dict = {
            "baseline_manifest_ref": {
                "run_timestamp": baseline_manifest.get("run_timestamp"),
                "sample_schema": baseline_manifest.get("sample_schema"),
            },
            "optimization_metric": metric,
            "min_precision": min_precision,
            "min_roc_auc": min_roc_auc,
            "tuning_n_iter": tuning_n_iter if tuning_n_iter is not None else 100,
            "manifest_path_used": resolved_manifest_path,
            "baseline_run_id": baseline_run_id,
            "tuning_time_limit_effective_minutes": effective_tuning_minutes,
            "tuning_time_limit_requested": time_limit_minutes,
        }
        med = getattr(pipeline.strategy, "monthly_median", None)
        if med is not None:
            try:
                merged_metrics[_STRATEGY_MONTHLY_CHARGES_MEDIAN_KEY] = float(med)
            except (TypeError, ValueError):
                pass
        merged_metrics.update(dict(pipeline.tuned_metrics))
        merged_metrics.update(dict(pipeline.guardrails_summary))
        if pipeline.best_model_name:
            merged_metrics["best_model_name"] = pipeline.best_model_name

        all_files: list[str] = []
        for root, _dirs, files in os.walk(run_root):
            for name in files:
                all_files.append(os.path.join(root, name))

        bundle_manifest_path = os.path.join(run_root, "manifest.json")
        man = build_manifest(
            pipeline_run_id=run.id,
            objective=objective,
            run_timestamp=run_ts,
            status="completed",
            input_filename=original_filename,
            input_path=baseline_input_path,
            paths_in_bundle=all_files,
            metrics=merged_metrics,
            mlflow_baseline_run_id=None,
            mlflow_fe_run_id=pipeline.mlflow_run_id,
            best_model_name=pipeline.best_model_name,
            original_filename=original_filename,
            active_deployment_id=active_dep,
        )
        write_manifest(man, bundle_manifest_path)
        man = build_manifest(
            pipeline_run_id=run.id,
            objective=objective,
            run_timestamp=run_ts,
            status="completed",
            input_filename=original_filename,
            input_path=baseline_input_path,
            paths_in_bundle=all_files + [bundle_manifest_path],
            metrics=merged_metrics,
            mlflow_baseline_run_id=None,
            mlflow_fe_run_id=pipeline.mlflow_run_id,
            best_model_name=pipeline.best_model_name,
            original_filename=original_filename,
            active_deployment_id=active_dep,
        )
        write_manifest(man, bundle_manifest_path)

        zip_path = os.path.join(tempfile.gettempdir(), f"fe_artifacts_{run.id}_{run_ts}.zip")
        if os.path.isfile(zip_path):
            safe_unlink(zip_path)
        zip_tree(run_root, zip_path)

        run.status = "completed"
        run.model_path = fe_joblib
        run.csv_output_path = None
        run.metrics = merged_metrics
        run.completed_at = utcnow()

    except Exception as e:
        logger.error(f"Feature Engineering falhou: {e}", exc_info=True)
        run.status = "failed"
        run.error_message = str(e)[:1000]
        run.completed_at = utcnow()
        run.active = False
    finally:
        if run_root:
            safe_rmtree(run_root)

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
