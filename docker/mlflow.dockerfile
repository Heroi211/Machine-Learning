FROM ghcr.io/mlflow/mlflow:v2.22.0

# Backend store PostgreSQL (bases airflow / processing / mlflow no db_processing).
# Cliente API usa mlflow 3.x (aliases); servidor 2.22+ expõe set_registered_model_alias.
RUN pip install --no-cache-dir psycopg2-binary
