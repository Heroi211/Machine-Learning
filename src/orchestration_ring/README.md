# orchestration_ring (anel 4)

**Orquestrador central de treino** — Airflow + integração DVC.

## Responsabilidade

- DAGs Airflow (`ml_training_pipeline`, `ml_training_dispatch`, drift, …)
- Tasks: dispatch por `domain`, `dvc repro`, persistência em `pipeline_runs`
- Variables / bootstrap JSON
- **Não** expõe API pública ao frontend

## Pode importar

- `ml_core_ring` (registries, hooks)
- `executors_ring` (invocação directa em tasks Python)
- `infra_ring`

## Não deve importar

- `platform_ring`

## Fluxo

```text
Airflow conf { "domain": "recommendation" }
  → get_domain → get_train_backend
  → executors_ring / dvc repro
  → grava pipeline_runs (BD partilhada com platform_ring)
```

## Código actual (transição)

| Origem | Destino |
|--------|---------|
| `airflow/dags/` | `orchestration_ring/airflow/dags/` (Fase 3) |
| `dvc.yaml` (raiz) | invocado por tasks aqui |

A API `platform_ring` dispara DAGs via REST (`trigger-dag`), não duplica lógica de treino.
