# Worker de recomendação — mesma base da API (Torch CPU + deps ML).
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY docker/requirements-api.txt ./requirements-api.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu \
    && pip install --no-cache-dir -r requirements-api.txt

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /var/www

COPY worker_recommendation.py ./worker_recommendation.py
COPY params.yaml ./params.yaml
COPY src ./src

EXPOSE 8010

CMD ["uvicorn", "worker_recommendation:app", "--host", "0.0.0.0", "--port", "8010"]
