# -----------------------------------------------------------------------------
# Multi-stage image: apenas main.py + src no layer final (ver .dockerignore no raiz).
# Torch via índice CPU oficial (bem menor que wheel com CUDA do PyPI).
# -----------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# netifaces pode precisar de compilar se não existir rodas para alvo; apenas nesta fase.
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

# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

LABEL maintainer="Gabriel Drumond <gabriel.drumond@cod3bit.com.br>"

# Runtime: cliente Postgres (drivers asyncpg já linkam só libc), OpenMP para numpy/sklearn/torch CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /var/www

COPY main.py ./main.py
COPY src ./src

EXPOSE 8000

# Evita reload e segundo processo watcher (principal.py usa reload=True apenas em modo dev direto).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
