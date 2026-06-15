# ml_core_ring (anel 2)

Contratos ML reutilizáveis — **registries, sem algoritmos pesados**.

## Responsabilidade

- `DomainPlugin` registry (`get_domain`)
- `TrainBackend` registry (`get_train_backend`) — Fase 1
- `InferenceEngine` registry (`get_engine`)
- `ArtifactManifest`, `PredictionResult`
- `PipelineRunner` (Template Method) — pipelines reprodutíveis
- `orchestration_hooks` — interface Airflow → backend — Fase 1

## Pode importar

- `infra_ring`

## Não deve importar

- `platform_ring`, `executors_ring`, `orchestration_ring`

## Código actual (transição)

| Origem | Nota |
|--------|------|
| `src/core/ml/` | Migrar para aqui na Fase 1 |

Ver também [PLANO_A_CORE_ML.md](../../docs/PLANO_A_CORE_ML.md).
