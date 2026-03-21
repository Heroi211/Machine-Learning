FROM postgres:16.2-alpine
LABEL maintainer "Gabriel Drumond <gabriel.drumond@cod3bit.com.br>"
ENV POSTGRES_USER=gabriel_drumond
ENV POSTGRES_PASSWORD=280387
ENV POSTGRES_DB=processing
EXPOSE 5432