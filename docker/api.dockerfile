FROM python:3.11-alpine
LABEL maintainer "Gabriel Drumond <gabriel.drumond@cod3bit.com.br>"
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
RUN apk update && apk upgrade && apk add postgresql-dev python3-dev --no-cache build-base freetds freetds-dev  openssl krb5-dev

COPY . /var/www
WORKDIR /var/www

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

EXPOSE 8000
ENTRYPOINT python main.py

