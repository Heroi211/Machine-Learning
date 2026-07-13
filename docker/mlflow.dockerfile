FROM ghcr.io/mlflow/mlflow:v2.22.0

# Backend store PostgreSQL (bases airflow / processing / mlflow no db_processing).
RUN pip install --no-cache-dir psycopg2-binary
