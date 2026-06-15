#!/usr/bin/env bash
# Chamado pelo contentor airflow-init após db migrate (ver docker-compose).
set -euo pipefail
CONF=/opt/airflow/bootstrap/ml_training_pipeline_conf.json
if [[ -f "$CONF" ]]; then
  airflow variables set ml_training_pipeline_conf "$(cat "$CONF")"
  echo "Variable ml_training_pipeline_conf definida a partir de $CONF"
else
  echo "Ficheiro $CONF ausente — Variable ml_training_pipeline_conf não alterada (configura na UI se precisares)."
fi

DRIFT=/opt/airflow/bootstrap/drift_monitoring_conf.json
if [[ -f "$DRIFT" ]]; then
  airflow variables set drift_monitoring_conf "$(cat "$DRIFT")"
  echo "Variable drift_monitoring_conf definida a partir de $DRIFT"
else
  echo "Ficheiro $DRIFT ausente — Variable drift_monitoring_conf não alterada."
fi

DISPATCH=/opt/airflow/bootstrap/ml_training_dispatch_conf.json
if [[ -f "$DISPATCH" ]]; then
  airflow variables set ml_training_dispatch_conf "$(cat "$DISPATCH")"
  echo "Variable ml_training_dispatch_conf definida a partir de $DISPATCH"
else
  echo "Ficheiro $DISPATCH ausente — Variable ml_training_dispatch_conf não alterada."
fi
