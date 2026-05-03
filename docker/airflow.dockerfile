# Imagem Apache Airflow com deps do projeto (tasks importam ``core``, ``services``, pipelines ML).
# Razão típica de falha sem isto: ``ModuleNotFoundError: pydantic_settings`` ao importar
# ``airflow_persistence``.

FROM apache/airflow:2.9.1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/airflow/ml_libs \
    && chown -R airflow:root /opt/airflow/ml_libs \
    && chmod 755 /opt/airflow/ml_libs

COPY docker/requirements-api.txt /opt/airflow/ml_libs-requirements.txt
RUN chown airflow:root /opt/airflow/ml_libs-requirements.txt

USER airflow
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalação isolada: não substitui Flask/SQLAlchemy/Werkzeug do próprio Airflow.
# O DAG faz ``sys.path.insert(0, "/opt/airflow/ml_libs")`` no início das tasks que importam ``services``.
RUN pip install --no-cache-dir \
        --target /opt/airflow/ml_libs \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu \
    && pip install --no-cache-dir --target /opt/airflow/ml_libs -r /opt/airflow/ml_libs-requirements.txt \
    && rm -f /opt/airflow/ml_libs-requirements.txt
