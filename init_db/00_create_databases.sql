-- Bases separadas no mesmo cluster Postgres (contentor db_processing).
-- processing: POSTGRES_DB (schema da app em database.sql)
-- airflow:    metadados Airflow (DAG runs, variables, …)
-- mlflow:     tracking + Model Registry (backend store do mlflow server)

CREATE DATABASE airflow;
CREATE DATABASE mlflow;
