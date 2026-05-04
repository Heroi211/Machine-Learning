FROM python:3.11-slim
LABEL maintainer "Gabriel Drumond <gabriel.drumond@cod3bit.com.br>"
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        postgresql-dev \
        python3-dev \
        build-essential \
        freetds-dev \
        openssl \
        krb5-dev

COPY . /var/www
WORKDIR /var/www

RUN pip install --upgrade pip
RUN pip install asyncpg
RUN pip install --no-cache-dir -r ./requirements.txt

EXPOSE 8000
ENTRYPOINT python main.py

