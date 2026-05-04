import logging
import math
import os
import json
import re
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.custom_logger import setup_pipeline_run_logging
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from services.utils import utcnow

logger = logging.getLogger(__name__)


def _recall_from_metrics(metrics: dict | None) -> float:
    try:
        return float((metrics or {}).get("test_recall", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _fe_competition_score_from_metrics(metrics: dict | None) -> float:
    """
    Score usado no desempate entre runs FE: **validação cruzada** da métrica de optimização
    (ex. ``cv_recall`` quando ``optimization_metric`` é ``recall``), não o valor no conjunto
    de teste — alinhado ao critério do tuning.
    """
    if not metrics:
        return float("-inf")
    opt = (metrics.get("optimization_metric") or "").strip().lower()
    if not opt:
        return float("-inf")
    for key in (f"cv_{opt}", "best_cv_score"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            continue
    return float("-inf")


def _fe_model_metric_pair(metrics: dict | None) -> tuple[str, str]:
    m = metrics or {}
    name = (m.get("best_model_name") or "").strip()
    opt = (m.get("optimization_metric") or "").strip().lower()
    return name, opt


async def _baseline_recall_winner(session: AsyncSession, run: PipelineRuns, run_timestamp: str) -> None:
    """
    Mantém no máximo um baseline ``active`` por ``objective``: compara ``test_recall`` (teste)
    com outros baselines ``completed`` e ``active``; se o novo for estritamente melhor,
    desactiva os anteriores, **publica** ``pre_processed/*`` partir do snapshot, e marca o novo ativo;
    caso contrário desactiva o novo (**sem** tocar no contrato FE global).

    ``run_timestamp``: pasta ``PATH_DATA/PATH_LOGS/<ts>`` com manifest/sample da corrida.
    """
    if run.pipeline_type != "baseline" or run.status != "completed":
        return

    new_recall = _recall_from_metrics(run.metrics)
    obj = (run.objective or "").strip().lower()

    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != run.id,
            PipelineRuns.pipeline_type == "baseline",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            PipelineRuns.is_airflow_run.is_(run.is_airflow_run),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    champions = list(res.scalars().all())

    if not champions:
        _publish_global_baseline_from_snapshot(run_timestamp)
        run.active = True
        logger.info(
            "Baseline run %s primeiro baseline ou sem campões recall — manifest global publicado.",
            run.id,
        )
        return

    best_prev = max((_recall_from_metrics(c.metrics) for c in champions), default=float("-inf"))

    if new_recall > best_prev:
        _publish_global_baseline_from_snapshot(run_timestamp)
        for c in champions:
            c.active = False
            session.add(c)
            logger.info(
                "Baseline run %s desactivado pelo comparador recall (mantido run %s, test_recall=%.6f > %.6f).",
                c.id,
                run.id,
                new_recall,
                best_prev,
            )
        run.active = True
    else:
        run.active = False
        logger.info(
            "Baseline run %s desactivado: test_recall=%.6f <= melhor anterior=%.6f.",
            run.id,
            new_recall,
            best_prev,
        )


async def _deactivate_other_fe_runs_for_objective(
    session: AsyncSession, *, objective: str, keep_run_id: int
) -> None:
    """Garante no máximo um FE `active` por domínio: desliga todos os outros concluídos."""
    obj = objective.strip().lower()
    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != keep_run_id,
            PipelineRuns.pipeline_type == "feature_engineering",
            PipelineRuns.active.is_(True),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    for other in res.scalars().all():
        other.active = False
        session.add(other)
        logger.info(
            "FE run %s desactivado (só um campeão activo por objective): cede lugar ao run %s.",
            other.id,
            keep_run_id,
        )


async def _fe_recall_winner(session: AsyncSession, run: PipelineRuns) -> None:
    """
    Mantém no máximo um FE ``active`` por ``objective`` quando há linhagem comparável:
    compara o **melhor score em CV** da ``optimization_metric`` (ex. ``cv_recall``), não métricas
    no holdout de teste; exige ``best_model_name`` e ``optimization_metric`` iguais a um campeão.
    Não altera ``pre_processed`` (contrato continua a ser do baseline).

    Após marcar um run como vencedor, **desactiva qualquer outro** FE activo do mesmo objective
    (independentemente de Airflow vs manual ou ``inference_backend``), para evitar dois ``active``
    simultâneos e confusão no promote/predict.
    """
    if run.pipeline_type != "feature_engineering" or run.status != "completed":
        return

    new_score = _fe_competition_score_from_metrics(run.metrics)
    new_name, new_metric = _fe_model_metric_pair(run.metrics)
    obj = (run.objective or "").strip().lower()

    res = await session.execute(
        select(PipelineRuns).where(
            PipelineRuns.id != run.id,
            PipelineRuns.pipeline_type == "feature_engineering",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            PipelineRuns.is_airflow_run.is_(run.is_airflow_run),
            PipelineRuns.inference_backend == (run.inference_backend or "sklearn"),
            func.lower(PipelineRuns.objective) == obj,
        )
    )
    champions = list(res.scalars().all())

    if not champions:
        await _deactivate_other_fe_runs_for_objective(session, objective=run.objective, keep_run_id=run.id)
        run.active = True
        logger.info(
            "FE run %s primeiro FE ou sem campeões activos no objective %r — marcado activo.",
            run.id,
            obj,
        )
        return

    compatible = [c for c in champions if _fe_model_metric_pair(c.metrics) == (new_name, new_metric)]
    if not compatible:
        run.active = False
        champ_summary = [
            {
                "id": c.id,
                "best_model_name": _fe_model_metric_pair(c.metrics)[0],
                "optimization_metric": _fe_model_metric_pair(c.metrics)[1],
            }
            for c in champions
        ]
        logger.warning(
            "FE run %s desactivado: best_model_name=%r ou optimization_metric=%r não coincidem com "
            "campeões activos %s. Requer validação manual antes de promover.",
            run.id,
            new_name,
            new_metric,
            champ_summary,
        )
        return

    best_prev = max((_fe_competition_score_from_metrics(c.metrics) for c in compatible), default=float("-inf"))

    if new_score > best_prev:
        await _deactivate_other_fe_runs_for_objective(session, objective=run.objective, keep_run_id=run.id)
        run.active = True
        logger.info(
            "FE run %s campeão cv_%s (score=%.6f > %.6f, modelo=%r métrica_optim=%r). Outros FE do objective desactivados.",
            run.id,
            new_metric,
            new_score,
            best_prev,
            new_name,
            new_metric,
        )
    else:
        run.active = False
        logger.info(
            "FE run %s desactivado: cv_%s=%.6f <= melhor anterior=%.6f (modelo=%r métrica_optim=%r).",
            run.id,
            new_metric,
            new_score,
            best_prev,
            new_name,
            new_metric,
        )


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


def _ts_from_baseline_model_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    m = re.search(r"baseline_model_[^_]+_(\d{8}_\d{6})\.joblib$", os.path.basename(model_path))
    return m.group(1) if m else None


def _global_preprocessed_manifest() -> str:
    return os.path.abspath(os.path.join(settings.path_data_preprocessed, "manifest.json"))


async def _resolve_fe_manifest(session: AsyncSession, objective: str) -> str:
    """
    1) Tenta ``pre_processed/manifest.json``.
    2) Se falhar: baseline ``completed``, ``active`` e mesmo ``objective`` na BD —
       manifest ao lado do ``baseline_sample`` do run ou em ``PATH_DATA/PATH_LOGS/<ts>/``.
    """
    obj = objective.strip().lower()

    cand_global = _global_preprocessed_manifest()
    if os.path.isfile(cand_global):
        logger.info("FE: usando manifest global em %s", cand_global)
        return cand_global

    stmt = (
        select(PipelineRuns)
        .where(
            PipelineRuns.pipeline_type == "baseline",
            PipelineRuns.status == "completed",
            PipelineRuns.active.is_(True),
            func.lower(PipelineRuns.objective) == obj,
        )
        .order_by(PipelineRuns.completed_at.desc(), PipelineRuns.id.desc())
        .limit(1)
    )
    res = await session.execute(stmt)
    run = res.scalars().one_or_none()
    if not run:
        raise ValueError(
            f"Manifest em falta: não há ``pre_processed/manifest.json`` e nenhum baseline activo para "
            f"objective={objective!r}. Rode baseline (API até ganhar recall ou modo sem defer)."
        )

    resolved: str | None = None

    outp = run.csv_output_path or ""
    if outp and os.path.isfile(outp):
        snap_manifest = os.path.join(os.path.dirname(os.path.abspath(outp)), "manifest.json")
        if os.path.isfile(snap_manifest):
            resolved = os.path.abspath(snap_manifest)

    if not resolved:
        ts = _ts_from_baseline_model_path(run.model_path)
        if ts:
            by_ts = os.path.abspath(
                os.path.join(settings.path_data, settings.path_logs, ts, "manifest.json")
            )
            if os.path.isfile(by_ts):
                resolved = by_ts

    if not resolved:
        raise FileNotFoundError(
            f"Baseline activo (pipeline_run_id={run.id}) sem manifest encontrado no disco. "
            f"Espere Paths ``csv_output_path`` / modelo com timestamp compatíveis com PATH_DATA/PATH_LOGS.",
        )

    logger.info(
        "FE: manifest global ausente — fallback pelo baseline activo (run_id=%s) em %s",
        run.id,
        resolved,
    )
    return resolved


async def _resolve_fe_manifest_isolated_session(objective: str) -> str:
    """Consulta rápida com sessão própria (antes de iniciar run FE na sessão principal)."""
    from core.database import Session as SessionFactory

    session = SessionFactory()
    try:
        path = await _resolve_fe_manifest(session, objective)
        await session.commit()
        return path
    finally:
        await session.close()


def _publish_global_baseline_from_snapshot(run_timestamp: str) -> None:
    """
    Copia ``baseline_sample`` + ``manifest.json`` do snapshot para ``pre_processed/``,
    ajustando no JSON o campo ``output_sample_csv_stable``.
    """
    snap_root = os.path.join(settings.path_data, settings.path_logs, run_timestamp.strip())
    src_manifest = os.path.join(snap_root, "manifest.json")
    src_sample = os.path.join(snap_root, "baseline_sample.csv")
    if not os.path.isfile(src_manifest):
        raise FileNotFoundError(f"Manifest snapshot inexistente: {src_manifest}")
    if not os.path.isfile(src_sample):
        raise FileNotFoundError(f"Sample baseline no snapshot inexistente: {src_sample}")

    os.makedirs(settings.path_data_preprocessed, exist_ok=True)
    dst_sample = os.path.abspath(os.path.join(settings.path_data_preprocessed, "baseline_sample.csv"))
    shutil.copy2(src_sample, dst_sample)

    with open(src_manifest, encoding="utf-8") as f:
        man = json.load(f)
    man["output_sample_csv_stable"] = dst_sample
    man["manifest_snapshot"] = os.path.abspath(src_manifest)
    dst_manifest = os.path.abspath(os.path.join(settings.path_data_preprocessed, "manifest.json"))
    with open(dst_manifest, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    logger.info("Contrato FE global atualizado a partir do snapshot %s.", snap_root)


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

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="baseline",
        objective=objective,
        status="processing",
        original_filename=file.filename,
        is_airflow_run=False,
    )

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
            from services.pipelines.binary_decision_threshold import labels_from_probability_threshold
            pipeline = Baseline(
                pobjective=objective,
                run_timestamp=run_ts,
                csv_path=input_path,
                class_labels=get_class_labels(objective),
                defer_global_preprocess_contract=True,
            )
            pipeline.run(start_time=datetime.now())
            pipeline.save_artifacts()

            model_path = os.path.join(settings.path_model, f"baseline_model_{objective}_{pipeline.now}.joblib")
            csv_path = os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)

            model = pipeline.model
            y_pred_test = labels_from_probability_threshold(model, pipeline.x_test, pipeline.decision_threshold)
            y_proba_test = model.predict_proba(pipeline.x_test)[:, 1]
            _zd = {"zero_division": 0}
            yt = pipeline.y_test
            metrics = {
                "classification_decision_threshold": float(pipeline.decision_threshold),
                "test_accuracy": float(accuracy_score(yt, y_pred_test)),
                "test_f1": float(f1_score(yt, y_pred_test, **_zd)),
                "test_precision": float(precision_score(yt, y_pred_test, **_zd)),
                "test_recall": float(recall_score(yt, y_pred_test, **_zd)),
            }
            if int(yt.sum()) > 0 and int(len(yt) - yt.sum()) > 0:
                metrics["test_pr_auc"] = float(average_precision_score(yt, y_proba_test))

            run.status = "completed"
            run.model_path = model_path
            run.csv_output_path = csv_path
            run.completed_at = utcnow()
            run.metrics = metrics

            await _baseline_recall_winner(session, run, run_ts)

            metrics["baseline_fe_contract_published"] = bool(run.active)
            run.metrics = metrics

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
    time_limit_minutes: int = 2,
    acc_target: float | None = None,
    decision_threshold: float | None = None,
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
    resolved_manifest_path = await _resolve_fe_manifest_isolated_session(objective)
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
    # Contrato FE: o treino segue sempre ``output_sample_csv_stable`` (ex. pre_processed/baseline_sample.csv).
    # Gravamos esse ficheiro como etiqueta na BD; o CSV “upstream” do baseline fica só nas métricas de auditoria.
    fe_training_csv_basename = os.path.basename(csv_baseline)
    original_filename = fe_training_csv_basename

    run = PipelineRuns(
        user_id=user_id,
        pipeline_type="feature_engineering",
        objective=objective,
        status="processing",
        original_filename=original_filename,
        is_airflow_run=False,
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
            decision_threshold=decision_threshold,
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
            "fe_training_csv": csv_baseline,
            "fe_training_csv_basename": fe_training_csv_basename,
            "baseline_upstream_input_basename": (
                os.path.basename(baseline_input_path) if baseline_input_path else None
            ),
            "baseline_upstream_input_path": baseline_input_path,
            "baseline_manifest_ref": {
                "run_timestamp": baseline_manifest.get("run_timestamp"),
                "sample_schema": baseline_manifest.get("sample_schema"),
            },
            "optimization_metric": metric,
            "min_precision": min_precision,
            "min_roc_auc": min_roc_auc,
            "tuning_n_iter": tuning_n_iter if tuning_n_iter is not None else 100,
            "manifest_path_used": resolved_manifest_path,
            "tuning_time_limit_effective_minutes": effective_tuning_minutes,
            "tuning_time_limit_requested": time_limit_minutes,
        }
        _bcs = float(pipeline.best_cv_score)
        merged_metrics["best_cv_score"] = _bcs
        if math.isfinite(_bcs):
            merged_metrics[f"cv_{metric}"] = _bcs
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
            input_path=csv_baseline,
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
            input_path=csv_baseline,
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

        backend = "mlp" if settings.use_mlp_for_prediction else "sklearn"
        merged_metrics["inference_backend"] = backend
        merged_metrics["predict_model"] = "pytorch_mlp" if backend == "mlp" else "sklearn_pipeline"
        merged_metrics["sklearn_benchmark_classifier"] = pipeline.best_model_name
        if backend == "mlp" and pipeline.mlp_artifact_prefix:
            merged_metrics["mlp_artifact_prefix"] = pipeline.mlp_artifact_prefix

        run.status = "completed"
        run.model_path = fe_joblib
        run.csv_output_path = None
        run.metrics = merged_metrics
        run.completed_at = utcnow()
        run.inference_backend = backend

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
        if merged.status == "completed" and merged.pipeline_type == "feature_engineering":
            await _fe_recall_winner(session, merged)
            m_final = dict(merged.metrics or {})
            m_final["fe_recall_champion"] = bool(merged.active)
            merged.metrics = m_final
            session.add(merged)
        await session.commit()
        await session.refresh(merged)

    return merged, zip_path


# Airflow (worker) e API (FastAPI) montam o **mesmo volume** ``ml_shared`` em mount points
# diferentes (``/opt/airflow/ml_project`` vs ``/var/www/ml_shared``). Caminhos gravados pelo
# Airflow precisam ser remapeados em runtime para ler do volume dentro da API.
_SHARED_PATH_REMAP: tuple[tuple[str, str], ...] = (
    ("/opt/airflow/ml_project/", "/var/www/ml_shared/"),
)


def _resolve_shared_artifact_path(p: str | None) -> str | None:
    """Traduz prefixos cross-container do volume partilhado ml_shared.

    Mantém ``None`` e caminhos já no formato local inalterados.
    """
    if not p:
        return p
    for src, dst in _SHARED_PATH_REMAP:
        if p.startswith(src):
            return dst + p[len(src):]
    return p


async def predict_for_domain(domain: str, features: dict, user_id: int, db: AsyncSession) -> Predictions:
    """Resolve o modelo ativo do domínio e prediz para um indivíduo.

    O ramo de inferência (sklearn vs MLP PyTorch) vem do campo ``inference_backend`` do
    ``PipelineRuns`` promovido — não da env. Assim, alternar ``USE_MLP_FOR_PREDICTION``
    afeta apenas **novos treinos/promotes**, não muda silenciosamente o que já está em
    produção.
    """
    from services.processor.deployment_service import NoActiveDeploymentError, get_active_deployment

    async with db as session:
        deployment = await get_active_deployment(domain, session)
        if not deployment:
            raise NoActiveDeploymentError(
                f"Nenhum modelo ativo para o domínio '{domain}'. Promova um pipeline concluído (admin)."
            )

        run = deployment.pipeline_run
        backend = (run.inference_backend or "sklearn").strip().lower()

        df_input = _prepare_prediction_features(run, domain, features)

        if backend == "mlp":
            from services.pipelines.mlp_inference import load_mlp_bundle, predict_with_mlp

            prefix = _resolve_shared_artifact_path((run.metrics or {}).get("mlp_artifact_prefix"))
            if not prefix:
                raise ValueError(
                    f"Run {run.id} promovido com inference_backend='mlp' mas sem 'mlp_artifact_prefix' "
                    "nas métricas. Treine o FE de novo com USE_MLP_FOR_PREDICTION=true."
                )
            bundle = load_mlp_bundle(prefix)
            label, prob = predict_with_mlp(bundle, df_input)
            prediction_value = int(label)
            probability = float(prob)
        else:
            local_model_path = _resolve_shared_artifact_path(run.model_path)
            if not local_model_path or not os.path.exists(local_model_path):
                raise ValueError(f"Modelo não encontrado em: {run.model_path}")

            model = joblib.load(local_model_path)
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
