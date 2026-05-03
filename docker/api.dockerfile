FROM python:3.11-slim-bookworm
LABEL maintainer "Gabriel Drumond <gabriel.drumond@cod3bit.com.br>"
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Debian (glibc): PyTorch e scipy têm wheels no PyPI; Alpine/musl não.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    freetds-dev \
    libkrb5-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /var/www
WORKDIR /var/www

RUN pip install --upgrade pip
RUN pip install asyncpg
RUN pip install --no-cache-dir -r ./requirements.txt

EXPOSE 8000
ENTRYPOINT python main.py
