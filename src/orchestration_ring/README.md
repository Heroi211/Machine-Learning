# orchestration_ring (anel 4)

**Orquestrador central de treino** — Airflow + integração DVC.

## Responsabilidade

- DAGs Airflow (`ml_training_dispatch`, `ml_training_pipeline`, drift, …)
- Tasks: dispatch por `domain`, persistência em `pipeline_runs`
- Variables / bootstrap JSON

## Módulos (Fase 3)

| Módulo | Papel |
|--------|-------|
| `conf.py` | Merge Variable + `dag_run.conf` |
| `dispatch.py` | Resolve domínio → rota tabular/recommendation |
| `dispatch_tasks.py` | Task branch `validate_dispatch` |
| `tabular_training.py` | Baseline + FE + promote |
| `recommendation_training.py` | Treino reco + persist |
| `persist_run.py` | BD partilhada (`pipeline_runs`) |
| `airflow_env.py` | Paths workers Airflow |

## DAG dispatch

```bash
airflow dags trigger ml_training_dispatch --conf '{"domain":"recommendation","top_k":10}'
airflow dags trigger ml_training_dispatch --conf '{"domain":"churn","csv_path":"/path/data.csv"}'
```

Variable: `ml_training_dispatch_conf` (bootstrap: `airflow/bootstrap/ml_training_dispatch_conf.json`).

## Pode importar

- `ml_core_ring`, `executors_ring`, `infra_ring`

## Não deve importar

- `platform_ring` (transição: `persist_run` delega tabular a `airflow_persistence`)

Ver [MIGRATION_RINGS.md](../../docs/MIGRATION_RINGS.md).
