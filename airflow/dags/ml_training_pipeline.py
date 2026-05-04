"""
DAG: ml_training_pipeline

Orquestra o fluxo completo de treino de um modelo de classificação binária:
  1. Validar CSV de entrada (colunas mínimas do domínio via ``strategy.validate``)
  2. Desactivar runs **manuais** na BD (``is_airflow_run=false``) do mesmo objective
  3. Executar Baseline (``defer_global_preprocess_contract`` alinhado à API) e persistir ``PipelineRuns``
  4. Executar Feature Engineering, persistir run com ``is_airflow_run=true`` e comparador cv_*
  5. Opcionalmente ``auto_promote`` (Variable/conf) chama o mesmo fluxo de promote da API
  6. Notificar conclusão (logs)

Parâmetros efetivos = merge(**defaults Airflow Variable**, **dag_run.conf**). O conf do trigger
(API ou UI) sobrescreve a Variable — útil em dev com API; em produção costuma-se deixar o
JSON na Variable e dar Trigger com conf vazio ou só overrides pontuais.

Variable (Admin → Variables), chave ``ml_training_pipeline_conf``:
    Valor = **uma única string JSON** com ``objective`` e ``csv_path`` obrigatórios nos runs.
    O merge com ``dag_run.conf`` do trigger faz o conf sobrescrever a Variable campo a campo.

    **JSON por omissão** (bootstrap: ``airflow/bootstrap/ml_training_pipeline_conf.json``)::

      {
        "objective": "churn",
        "csv_path": "/opt/airflow/ml_project/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "optimization_metric": "recall",
        "time_limit_minutes": 30,
        "tuning_n_iter": 1000,
        "user_id": 2,
        "auto_promote": false
      }

    **Campos (resumo)**
      - ``objective``: nome do domínio registado nas strategies (ex. ``churn``).
      - ``csv_path``: caminho absoluto no worker (volume ``ml_project`` / uploads).
      - ``optimization_metric``: métrica que guia ranking e tuning no FE (ex. ``recall``, ``accuracy``).
      - ``time_limit_minutes``: minutos máximos do passo de tuning (time-box).
      - ``tuning_n_iter``: número máximo de configs amostradas no tuning antes do time-box.
      - ``user_id``: ``users.id`` na BD da aplicação para o ``PipelineRuns``; omitir usa
        ``DEFAULT_PIPELINE_USER_ID`` (por omissão 2 via env ``ML_PIPELINE_DEFAULT_USER_ID``).
      - ``auto_promote``: se ``true``, após FE executa fluxo equivalente ao promote da API.

    **Opcionais** (omitir ou não incluir a chave = sem guardrail / omissão do código)
      - ``min_precision``: mínimo de **precision média em CV** para contar candidato que passa
        guardrails na comparação e no tuning (endurece bastante quando se optimiza ``recall``).
      - ``min_roc_auc``: idem para **ROC AUC médio em CV**.
      - ``acc_target``: número alvo comunicado ao pipeline no final do tuning (reporte/logs).
      - ``decision_threshold``: probabilidade mínima P(churn) para classificar positivo nas métricas
        de teste (default vê env ``CLASSIFICATION_DECISION_THRESHOLD``, tipicamente 0.5; ex. 0.25
        para priorizar recall em relatórios). Não altera os scores de CV do tuning.

      Exemplo **com guardrails restritivos** (só se fizer sentido ao negócio)::

      {
        "objective": "churn",
        "csv_path": "/opt/airflow/ml_project/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "optimization_metric": "recall",
        "time_limit_minutes": 30,
        "min_precision": 0.7,
        "min_roc_auc": 0.75,
        "tuning_n_iter": 1000,
        "user_id": 2,
        "auto_promote": false
      }

    Texto pensado para o campo **Description** da Variable na UI (copy-paste):
    ``airflow/bootstrap/ml_training_pipeline_conf.variable_description.txt``

    Opcionalmente ``ml_training_objective`` e ``ml_training_csv_path`` (Variáveis string)
    quando o JSON não traz apenas esses dois campos — preenchem lacunas antes do merge.

Disparo via CLI (conf explícito):
  airflow dags trigger ml_training_pipeline --conf '{ ... }'
"""
from __future__ import annotations

import json
import logging
import os
import sys

from pathlib import Path

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

ML_PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
ML_CODE_ROOT = os.environ.get("ML_CODE_ROOT", "").strip()
# ``users.id`` na BD da aplicação quando Variable/conf não incluem ``user_id`` (ex.: admin).
DEFAULT_PIPELINE_USER_ID = int(os.environ.get("ML_PIPELINE_DEFAULT_USER_ID", "2"))


def _bootstrap_ml_sys_path() -> None:
    """Imports ``services`` / ``core`` vivem em ``src/`` (montado em ``ML_CODE_ROOT`` no compose)."""
    code_root = ML_CODE_ROOT
    if not code_root:
        _candidate = Path(__file__).resolve().parents[2] / "src"
        if _candidate.is_dir():
            code_root = str(_candidate)
    for p in (ML_PROJECT_ROOT, code_root):
        if p and p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_ml_sys_path()


def _prepend_airflow_ml_site_packages() -> None:
    """
    Bibliotecas do projeto vivem em ``/opt/airflow/ml_libs`` (``pip install --target`` na imagem
    ``docker/airflow.dockerfile``) para não substituir as versões que o Apache Airflow trava nos
    mesmos pacotes no site-packages principal. Chamada só no arranque das tasks que importam
    ``services`` / ``core`` — não ao carregar este módulo, para não afectar o scheduler.
    """
    root = os.environ.get("ML_AIRFLOW_SITE_PACKAGES", "/opt/airflow/ml_libs").strip()
    if root and os.path.isdir(root):
        rp = os.path.abspath(root)
        if rp not in sys.path:
            sys.path.insert(0, rp)


DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _resolve_ml_artifact_path(path: str) -> str:
    """
    Caminhos gravados pelo Baseline costumam ser relativos ao diretório do projeto ML
    (ex.: ``data/pre_processed/manifest.json``). O worker do Airflow pode ter CWD
    diferente; tenta ``ML_PROJECT_ROOT`` antes de ``abspath`` isolado.
    """
    path = os.path.normpath(path.strip())
    if os.path.isfile(path):
        return os.path.abspath(path)
    under_root = os.path.join(ML_PROJECT_ROOT, path)
    if os.path.isfile(under_root):
        return os.path.abspath(under_root)
    abs_cwd = os.path.abspath(path)
    if os.path.isfile(abs_cwd):
        return abs_cwd
    return os.path.abspath(under_root)


def _merged_run_conf(context: dict) -> dict:
    """
    Defaults de Airflow Variables + overrides em dag_run.conf (trigger API/UI/CLI).
    Quem vem por último no merge vence: conf do run sobrescreve a Variable.
    """
    from airflow.models import Variable

    defaults: dict = {}
    try:
        raw = Variable.get("ml_training_pipeline_conf", default_var=None)
        if raw:
            defaults = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Airflow Variable 'ml_training_pipeline_conf' deve ser um JSON válido "
            f"(objective, csv_path, ...): {e}"
        ) from e
    except Exception as e:
        log.warning("Could not read ml_training_pipeline_conf: %s", e)

    for key, var_key in (
        ("objective", "ml_training_objective"),
        ("csv_path", "ml_training_csv_path"),
    ):
        if defaults.get(key):
            continue
        try:
            v = Variable.get(var_key, default_var=None)
            if v:
                defaults[key] = v
        except Exception:
            pass

    conf_run = context["dag_run"].conf or {}
    merged = {**defaults, **conf_run}
    log.info(
        "Conf efetiva: chaves=%s (Variable + conf do run; conf do run tem prioridade)",
        list(merged.keys()),
    )
    return merged


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def task_validate_input(**context) -> None:
    """Valida se o CSV existe e tem as colunas mínimas para o domínio (via ``strategy.validate``)."""
    _prepend_airflow_ml_site_packages()
    from core.configs import settings as _svc_settings

    conf = _merged_run_conf(context)
    objective = conf.get("objective")
    csv_path = conf.get("csv_path")

    if not objective:
        raise ValueError("Parâmetro 'objective' é obrigatório no dag_run.conf.")
    if not csv_path:
        raise ValueError("Parâmetro 'csv_path' é obrigatório no dag_run.conf.")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, nrows=5)
        log.info("CSV validado — shape preview: %s colunas | arquivo: %s", len(df.columns), csv_path)
    except Exception as e:
        raise ValueError(f"Erro ao ler o CSV: {e}")

    try:
        from services.pipelines.feature_strategies import STRATEGY_REGISTRY

        if objective not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Domínio '{objective}' não registrado. Disponíveis: {available}")
        strategy = STRATEGY_REGISTRY[objective]()
        strategy.validate(df)
        log.info("Validação de colunas obrigatórias OK para domínio '%s'.", objective)
    except ImportError:
        log.warning("STRATEGY_REGISTRY não acessível neste worker — validação de colunas pulada.")

    ti = context["task_instance"]
    ti.xcom_push(key="objective", value=objective)
    ti.xcom_push(key="csv_path", value=csv_path)
    ti.xcom_push(key="user_id", value=int(conf.get("user_id", DEFAULT_PIPELINE_USER_ID)))
    ti.xcom_push(key="optimization_metric", value=conf.get("optimization_metric", "accuracy"))
    ti.xcom_push(key="time_limit_minutes", value=int(conf.get("time_limit_minutes", 2)))
    ti.xcom_push(key="acc_target", value=float(conf.get("acc_target", 0.90)))

    if conf.get("min_precision") is not None:
        ti.xcom_push(key="min_precision", value=float(conf["min_precision"]))
    if conf.get("min_roc_auc") is not None:
        ti.xcom_push(key="min_roc_auc", value=float(conf["min_roc_auc"]))
    if conf.get("tuning_n_iter") is not None:
        ti.xcom_push(key="tuning_n_iter", value=int(conf["tuning_n_iter"]))

    ti.xcom_push(key="auto_promote", value=bool(conf.get("auto_promote", False)))

    _dt_raw = conf.get("decision_threshold")
    ti.xcom_push(
        key="decision_threshold",
        value=float(_svc_settings.classification_decision_threshold if _dt_raw is None else _dt_raw),
    )

    log.info("Validação de input OK para domínio '%s'.", objective)
    log.info(f"CSV path: {csv_path}")
    log.info(f"User ID: {int(conf.get('user_id', DEFAULT_PIPELINE_USER_ID))}")
    log.info(f"Optimization metric: {conf.get('optimization_metric', 'accuracy')}")
    log.info(f"Time limit minutes: {int(conf.get('time_limit_minutes', 2))}")
    log.info(f"Acc target: {float(conf.get('acc_target', 0.90))}")
    if conf.get("min_precision") is not None:
        log.info("min_precision: %s", conf["min_precision"])
    if conf.get("min_roc_auc") is not None:
        log.info("min_roc_auc: %s", conf["min_roc_auc"])
    if conf.get("tuning_n_iter") is not None:
        log.info("tuning_n_iter: %s", conf["tuning_n_iter"])
    log.info("auto_promote: %s", conf.get("auto_promote", False))
    log.info(
        "decision_threshold: %s",
        float(_svc_settings.classification_decision_threshold if _dt_raw is None else _dt_raw),
    )


def task_deactivate_manual_runs(**context) -> None:
    """Desactiva na BD runs baseline/FE manuais (``is_airflow_run=false``) do objective."""
    _prepend_airflow_ml_site_packages()
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    from services.processor.airflow_persistence import (
        deactivate_manual_pipeline_runs_for_objective,
        run_async,
    )

    run_async(deactivate_manual_pipeline_runs_for_objective(objective))


def task_run_baseline(**context) -> None:
    """Executa o pipeline Baseline e persiste o pipeline_run no banco."""
    _prepend_airflow_ml_site_packages()
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    csv_path = ti.xcom_pull(key="csv_path", task_ids="validate_input")
    user_id = ti.xcom_pull(key="user_id", task_ids="validate_input")
    decision_threshold = float(ti.xcom_pull(key="decision_threshold", task_ids="validate_input"))

    log.info(
        "Iniciando Baseline | objective=%s | csv=%s | decision_threshold=%s",
        objective,
        csv_path,
        decision_threshold,
    )

    try:
        from core.custom_logger import setup_pipeline_run_logging
        from services.pipelines.baseline import Baseline
        from services.pipelines.feature_strategies import get_class_labels

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(ML_PROJECT_ROOT, "logs", now)
        # Não usar ``setup_log`` aqui: ele faz ``root.handlers.clear()`` e remove os handlers
        # que o Airflow instala no worker, o que esconde tracebacks e pode gerar zombie tasks.
        setup_pipeline_run_logging(
            snapshot_path,
            now,
            run_id=None,
            objective=objective,
            pipeline_type="baseline",
        )

        pipeline = Baseline(
            pobjective=objective,
            run_timestamp=now,
            csv_path=csv_path,
            class_labels=get_class_labels(objective),
            defer_global_preprocess_contract=True,
            decision_threshold=decision_threshold,
        )
        pipeline.run(start_time=datetime.now())
        pipeline.save_artifacts()

        if pipeline.defer_global_preprocess_contract:
            baseline_manifest_path = os.path.abspath(
                os.path.join(pipeline.snapshot_path, pipeline.contract_manifest_name)
            )
            baseline_sample_csv_path = os.path.abspath(
                os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)
            )
        else:
            manifest_rel = os.path.join(pipeline.path_data_preprocessed, pipeline.contract_manifest_name)
            sample_rel = os.path.join(pipeline.path_data_preprocessed, pipeline.contract_sample_name)
            baseline_manifest_path = _resolve_ml_artifact_path(manifest_rel)
            baseline_sample_csv_path = _resolve_ml_artifact_path(sample_rel)

        if not os.path.isfile(baseline_manifest_path):
            raise FileNotFoundError(
                f"Manifest do baseline não encontrado após save (tentado: {baseline_manifest_path}). "
                f"CWD={os.getcwd()} ML_PROJECT_ROOT={ML_PROJECT_ROOT}"
            )
        if not os.path.isfile(baseline_sample_csv_path):
            raise FileNotFoundError(
                f"CSV estável do baseline não encontrado após save (tentado: {baseline_sample_csv_path})."
            )

        from services.processor.airflow_persistence import persist_airflow_baseline_run, run_async

        br_id = run_async(
            persist_airflow_baseline_run(
                objective=objective,
                user_id=int(user_id),
                run_ts=now,
                pipeline=pipeline,
                original_filename=os.path.basename(csv_path),
                airflow_dag_run_id=context["dag_run"].run_id,
            )
        )

        ti.xcom_push(key="baseline_manifest_path", value=baseline_manifest_path)
        ti.xcom_push(key="baseline_sample_csv_path", value=baseline_sample_csv_path)
        ti.xcom_push(key="baseline_run_ts", value=now)
        ti.xcom_push(key="baseline_pipeline_run_id", value=br_id)
        log.info(
            "Baseline concluído | manifest=%s | sample_csv=%s | pipeline_run_id=%s",
            baseline_manifest_path,
            baseline_sample_csv_path,
            br_id,
        )

    except ImportError as e:
        raise RuntimeError(f"Dependência ML não disponível no worker Airflow: {e}") from e
    except Exception:
        log.exception("run_baseline falhou (ver também pipeline_*.txt no snapshot em ml_project/logs).")
        raise


def task_run_fe(**context) -> None:
    """Executa o pipeline Feature Engineering e persiste o pipeline_run no banco."""
    _prepend_airflow_ml_site_packages()
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    optimization_metric = ti.xcom_pull(key="optimization_metric", task_ids="validate_input")
    time_limit_minutes = int(ti.xcom_pull(key="time_limit_minutes", task_ids="validate_input"))
    acc_target = ti.xcom_pull(key="acc_target", task_ids="validate_input")
    manifest_path = ti.xcom_pull(key="baseline_manifest_path", task_ids="run_baseline")
    user_id = ti.xcom_pull(key="user_id", task_ids="validate_input")
    min_precision = ti.xcom_pull(key="min_precision", task_ids="validate_input")
    min_roc_auc = ti.xcom_pull(key="min_roc_auc", task_ids="validate_input")
    tuning_n_iter = ti.xcom_pull(key="tuning_n_iter", task_ids="validate_input")
    decision_threshold = float(ti.xcom_pull(key="decision_threshold", task_ids="validate_input"))

    if not manifest_path or not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest do Baseline não encontrado para o FE: {manifest_path}")

    log.info(
        "Iniciando Feature Engineering | objective=%s | metric=%s | manifest=%s | "
        "min_precision=%s min_roc_auc=%s tuning_n_iter=%s decision_threshold=%s",
        objective,
        optimization_metric,
        manifest_path,
        min_precision,
        min_roc_auc,
        tuning_n_iter,
        decision_threshold,
    )

    try:
        from services.pipelines.feature_engineering import FeatureEngineering
        from services.pipelines.feature_strategies import STRATEGY_REGISTRY
        from services.processor.airflow_persistence import (
            persist_airflow_feature_engineering_run,
            run_async,
        )

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = STRATEGY_REGISTRY[objective]()
        pipeline = FeatureEngineering(
            objective=objective,
            strategy=strategy,
            run_timestamp=now,
            csv_path=None,
            manifest_path=manifest_path,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            decision_threshold=decision_threshold,
        )
        # Airflow: sem teto SYNC_FE — usa o valor integral do conf/Variable
        pipeline.run(time_limit_minutes=time_limit_minutes, acc_target=acc_target)

        fe_id, champion = run_async(
            persist_airflow_feature_engineering_run(
                objective=objective,
                user_id=int(user_id),
                run_ts=now,
                manifest_path=manifest_path,
                pipeline=pipeline,
                optimization_metric=optimization_metric,
                min_precision=min_precision,
                min_roc_auc=min_roc_auc,
                tuning_n_iter=tuning_n_iter,
                time_limit_minutes=time_limit_minutes,
                effective_tuning_minutes=time_limit_minutes,
                airflow_dag_run_id=context["dag_run"].run_id,
            )
        )

        ti.xcom_push(key="fe_run_ts", value=now)
        ti.xcom_push(key="fe_best_model", value=pipeline.best_model_name)
        ti.xcom_push(key="fe_metrics", value=str(pipeline.tuned_metrics))
        ti.xcom_push(key="fe_pipeline_run_id", value=fe_id)
        ti.xcom_push(key="fe_recall_champion", value=champion)
        log.info(
            "Feature Engineering concluído | melhor modelo: %s | pipeline_run_id=%s | campeão=%s",
            pipeline.best_model_name,
            fe_id,
            champion,
        )

    except ImportError as e:
        raise RuntimeError(f"Dependência ML não disponível no worker Airflow: {e}") from e
    except Exception:
        log.exception("run_fe falhou.")
        raise


def task_promote_fe_optional(**context) -> None:
    """Se ``auto_promote`` e o FE for campeão, promove o único FE activo (mesma lógica da API)."""
    _prepend_airflow_ml_site_packages()
    ti = context["task_instance"]
    if not ti.xcom_pull(key="auto_promote", task_ids="validate_input"):
        log.info("auto_promote=false — promote automático não executado.")
        return
    if not ti.xcom_pull(key="fe_recall_champion", task_ids="run_fe"):
        log.info("FE não venceu o comparador cv_* — promote automático não executado.")
        return
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    user_id = int(ti.xcom_pull(key="user_id", task_ids="validate_input"))
    from services.processor.airflow_persistence import promote_airflow_fe_if_requested, run_async

    run_async(
        promote_airflow_fe_if_requested(
            objective=objective,
            user_id=user_id,
            auto_promote=True,
        )
    )


def task_notify_complete(**context) -> None:
    """Loga os run_ids e métricas para que o admin decida sobre promoção."""
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    fe_best_model = ti.xcom_pull(key="fe_best_model", task_ids="run_fe")
    fe_metrics = ti.xcom_pull(key="fe_metrics", task_ids="run_fe")
    fe_run_ts = ti.xcom_pull(key="fe_run_ts", task_ids="run_fe")
    baseline_run_ts = ti.xcom_pull(key="baseline_run_ts", task_ids="run_baseline")
    bl_id = ti.xcom_pull(key="baseline_pipeline_run_id", task_ids="run_baseline")
    fe_id = ti.xcom_pull(key="fe_pipeline_run_id", task_ids="run_fe")
    champion = ti.xcom_pull(key="fe_recall_champion", task_ids="run_fe")

    log.info("=" * 60)
    log.info("PIPELINE CONCLUÍDO")
    log.info("Domínio               : %s", objective)
    log.info("Baseline pipeline_run : %s (run_ts=%s)", bl_id, baseline_run_ts)
    log.info("FE pipeline_run       : %s (run_ts=%s)", fe_id, fe_run_ts)
    log.info("FE campeão (cv_*)    : %s", champion)
    log.info("Melhor modelo FE      : %s", fe_best_model)
    log.info("Métricas FE           : %s", fe_metrics)
    if ti.xcom_pull(key="auto_promote", task_ids="validate_input"):
        log.info("auto_promote estava ligado — promote tentado na task anterior se campeão.")
    else:
        log.info("Promoção manual: POST /v1/processor/admin/promote (OBJECTIVE na env)")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="ml_training_pipeline",
    description="Treino completo: Baseline → Feature Engineering → notificação para promoção.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training", "classification"],
) as dag:

    validate = PythonOperator(task_id="validate_input", python_callable=task_validate_input)
    deactivate_manual = PythonOperator(
        task_id="deactivate_manual_runs",
        python_callable=task_deactivate_manual_runs,
    )
    baseline = PythonOperator(task_id="run_baseline", python_callable=task_run_baseline)
    fe = PythonOperator(task_id="run_fe", python_callable=task_run_fe)
    promote = PythonOperator(task_id="promote_fe_optional", python_callable=task_promote_fe_optional)
    notify = PythonOperator(task_id="notify_complete", python_callable=task_notify_complete)

    validate >> deactivate_manual >> baseline >> fe >> promote >> notify
