"""
DAG: ml_training_pipeline

Orquestra o fluxo completo de treino de um modelo de classificação binária:
  1. Validar CSV de entrada (colunas mínimas do domínio)
  2. Executar pipeline Baseline
  3. Executar pipeline Feature Engineering
  4. Notificar conclusão com run_ids para decisão de promoção

Disparo manual (via Swagger do Airflow ou trigger-dag da API):
  airflow dags trigger ml_training_pipeline --conf '{
    "objective": "heart_disease",
    "csv_path": "/opt/airflow/ml_project/uploads/dados.csv",
    "optimization_metric": "recall",
    "time_limit_minutes": 30,
    "acc_target": 0.85,
    "user_id": 1
  }'
"""
from __future__ import annotations

import logging
import os
import sys

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

ML_PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
if ML_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, ML_PROJECT_ROOT)

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def task_validate_input(**context) -> None:
    """Valida se o CSV existe e tem as colunas mínimas para o domínio."""
    conf = context["dag_run"].conf or {}
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

    context["task_instance"].xcom_push(key="objective", value=objective)
    context["task_instance"].xcom_push(key="csv_path", value=csv_path)
    context["task_instance"].xcom_push(key="user_id", value=conf.get("user_id", 1))
    context["task_instance"].xcom_push(key="optimization_metric", value=conf.get("optimization_metric", "accuracy"))
    context["task_instance"].xcom_push(key="time_limit_minutes", value=int(conf.get("time_limit_minutes", 2)))
    context["task_instance"].xcom_push(key="acc_target", value=float(conf.get("acc_target", 0.90)))


def task_run_baseline(**context) -> None:
    """Executa o pipeline Baseline e persiste o pipeline_run no banco."""
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    csv_path = ti.xcom_pull(key="csv_path", task_ids="validate_input")
    user_id = ti.xcom_pull(key="user_id", task_ids="validate_input")

    log.info("Iniciando Baseline | objective=%s | csv=%s", objective, csv_path)

    try:
        from core.custom_logger import setup_log
        from services.pipelines.baseline import Baseline
        from services.pipelines.feature_strategies import get_class_labels

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(ML_PROJECT_ROOT, "logs", now)
        setup_log(snapshot_path, now)

        pipeline = Baseline(
            pobjective=objective,
            run_timestamp=now,
            csv_path=csv_path,
            class_labels=get_class_labels(objective),
        )
        pipeline.run(start_time=datetime.now())
        pipeline.save_artifacts()

        csv_output = os.path.join(ML_PROJECT_ROOT, "data", "pre_processed", f"{objective}_sample_{now}.csv")
        ti.xcom_push(key="baseline_csv_output", value=csv_output)
        ti.xcom_push(key="baseline_run_ts", value=now)
        log.info("Baseline concluído | csv_output=%s", csv_output)

    except ImportError as e:
        raise RuntimeError(f"Dependência ML não disponível no worker Airflow: {e}")


def task_run_fe(**context) -> None:
    """Executa o pipeline Feature Engineering e persiste o pipeline_run no banco."""
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    optimization_metric = ti.xcom_pull(key="optimization_metric", task_ids="validate_input")
    time_limit_minutes = ti.xcom_pull(key="time_limit_minutes", task_ids="validate_input")
    acc_target = ti.xcom_pull(key="acc_target", task_ids="validate_input")
    csv_path = ti.xcom_pull(key="baseline_csv_output", task_ids="run_baseline")

    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV pré-processado do Baseline não encontrado: {csv_path}")

    log.info("Iniciando Feature Engineering | objective=%s | metric=%s | csv=%s", objective, optimization_metric, csv_path)

    try:
        from services.pipelines.feature_engineering import FeatureEngineering
        from services.pipelines.feature_strategies import STRATEGY_REGISTRY

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = STRATEGY_REGISTRY[objective]()
        pipeline = FeatureEngineering(objective=objective, strategy=strategy, run_timestamp=now, csv_path=csv_path, optimization_metric=optimization_metric)
        pipeline.run(time_limit_minutes=time_limit_minutes, acc_target=acc_target)

        ti.xcom_push(key="fe_run_ts", value=now)
        ti.xcom_push(key="fe_best_model", value=pipeline.best_model_name)
        ti.xcom_push(key="fe_metrics", value=str(pipeline.tuned_metrics))
        log.info("Feature Engineering concluído | melhor modelo: %s", pipeline.best_model_name)

    except ImportError as e:
        raise RuntimeError(f"Dependência ML não disponível no worker Airflow: {e}")


def task_notify_complete(**context) -> None:
    """Loga os run_ids e métricas para que o admin decida sobre promoção."""
    ti = context["task_instance"]
    objective = ti.xcom_pull(key="objective", task_ids="validate_input")
    fe_best_model = ti.xcom_pull(key="fe_best_model", task_ids="run_fe")
    fe_metrics = ti.xcom_pull(key="fe_metrics", task_ids="run_fe")
    fe_run_ts = ti.xcom_pull(key="fe_run_ts", task_ids="run_fe")
    baseline_run_ts = ti.xcom_pull(key="baseline_run_ts", task_ids="run_baseline")

    log.info("=" * 60)
    log.info("PIPELINE CONCLUÍDO")
    log.info("Domínio          : %s", objective)
    log.info("Baseline run_ts  : %s", baseline_run_ts)
    log.info("FE run_ts        : %s", fe_run_ts)
    log.info("Melhor modelo FE : %s", fe_best_model)
    log.info("Métricas FE      : %s", fe_metrics)
    log.info("Próximo passo: consultar pipeline_run_id no banco e promover via")
    log.info("  POST /v1/processor/admin/promote")
    log.info("  { 'domain': '%s', 'pipeline_run_id': <ID> }", objective)
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
    baseline = PythonOperator(task_id="run_baseline", python_callable=task_run_baseline)
    fe = PythonOperator(task_id="run_fe", python_callable=task_run_fe)
    notify = PythonOperator(task_id="notify_complete", python_callable=task_notify_complete)

    validate >> baseline >> fe >> notify
