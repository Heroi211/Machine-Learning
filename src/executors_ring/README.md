# executors_ring (anel 3)

Implementação de **treino e modelos** por tipo de problema.

## Responsabilidade

- `tabular_classification/` — Baseline, Feature Engineering, strategies, MLP inference, `TorchBundleEngine`
- `recommendation/` — facade pipeline DVC, `RecommendationDvcTrainBackend`, worker stub
- `TrainBackend` concretos registados em `ml_core_ring` (import side-effect)

## Pode importar

- `ml_core_ring`, `domains_ring`, `infra_ring`

## Não deve importar

- `platform_ring`, `orchestration_ring`

## Estrutura (Fase 2)

| Módulo | Papel |
|--------|-------|
| `tabular_classification/baseline.py` | Facade → `services.pipelines.baseline` |
| `tabular_classification/feature_engineering.py` | Facade → `services.pipelines.feature_engineering` |
| `tabular_classification/runner.py` | Execução programática Baseline + FE |
| `tabular_classification/train_backend.py` | `legacy_fe` backend |
| `tabular_classification/torch_bundle_engine.py` | Engine inferência MLP |
| `recommendation/pipeline_runner.py` | Facade → `domains.recommendation.pipeline_runner` |
| `recommendation/train_backend.py` | `recommendation_dvc` backend |
| `recommendation/worker.py` | Entrypoint worker (Fase 5: container HTTP) |

## Bootstrap

```python
import executors_ring  # registra TrainBackend + TorchBundleEngine
```

Ver [MIGRATION_RINGS.md](../../docs/MIGRATION_RINGS.md).
