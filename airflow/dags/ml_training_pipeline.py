"""
DAG: ml_training_pipeline

Orquestra treino tabular (Baseline → FE → promote opcional).
Mantido por compatibilidade; preferir ``ml_training_dispatch`` com ``domain`` no conf.

Tasks delegadas a ``orchestration_ring.tabular_training`` (Fase 3).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from orchestration_ring.airflow_env import bootstrap_ml_sys_path, prepend_airflow_ml_site_packages

bootstrap_ml_sys_path()

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _wrap(fn):
    def _inner(**context):
        prepend_airflow_ml_site_packages()
        return fn(**context)

    return _inner


with DAG(
    dag_id="ml_training_pipeline",
    description="Treino tabular legado: Baseline → Feature Engineering → promote opcional.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training", "classification"],
) as dag:
    from orchestration_ring.tabular_training import (
        task_deactivate_manual_runs,
        task_notify_tabular_complete,
        task_promote_fe_optional,
        task_run_baseline,
        task_run_fe,
        task_validate_tabular_input,
    )

    validate = PythonOperator(task_id="validate_input", python_callable=_wrap(task_validate_tabular_input))
    deactivate_manual = PythonOperator(
        task_id="deactivate_manual_runs",
        python_callable=_wrap(task_deactivate_manual_runs),
    )
    baseline = PythonOperator(task_id="run_baseline", python_callable=_wrap(task_run_baseline))
    fe = PythonOperator(task_id="run_fe", python_callable=_wrap(task_run_fe))
    promote = PythonOperator(task_id="promote_fe_optional", python_callable=_wrap(task_promote_fe_optional))
    notify = PythonOperator(
        task_id="notify_complete",
        python_callable=_wrap(task_notify_tabular_complete),
    )

    validate >> deactivate_manual >> baseline >> fe >> promote >> notify
