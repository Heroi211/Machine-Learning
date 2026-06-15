# executors_ring (anel 3)

Implementação de **treino e modelos** por tipo de problema.

## Responsabilidade

- `tabular_classification/` — Baseline, Feature Engineering, strategies tabulares
- `recommendation/` — pipeline DVC, embedding PyTorch, baselines sklearn
- `TrainBackend` concretos registados em `ml_core_ring`

## Pode importar

- `ml_core_ring`, `domains_ring`, `infra_ring`

## Não deve importar

- `platform_ring`, `orchestration_ring`

## Código actual (transição)

| Origem | Destino |
|--------|---------|
| `src/services/pipelines/` | `tabular_classification/` |
| `src/domains/recommendation/` (runner, models, stages) | `recommendation/` |

Workers Docker (`worker_app.py`) vivem aqui.
