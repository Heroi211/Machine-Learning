# Imagem Apache Airflow com deps do projeto (tasks importam ``core``, ``services``, pipelines ML).
# Razão típica de falha sem isto: ``ModuleNotFoundError: pydantic_settings`` ao importar
# ``airflow_persistence``.

FROM apache/airflow:2.9.1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu \
    && pip install --no-cache-dir -r /tmp/requirements-api.txt \
    && rm -f /tmp/requirements-api.txt
