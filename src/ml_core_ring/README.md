# ml_core_ring (anel 2)

Contratos ML reutilizáveis — **registries, sem algoritmos pesados**.

## Responsabilidade

- `DomainPlugin` registry (`get_domain`)
- `TrainBackend` registry (`get_train_backend`) — Fase 1
- `InferenceEngine` registry (`get_engine`)
- `ArtifactManifest`, `PredictionResult`
- `PipelineRunner` (Template Method) — pipelines reprodutíveis
- `orchestration_hooks` — interface Airflow → backend — Fase 1
- `mlflow_setup.py` — tracking URI + experimentos (Postgres `mlflow`, artefactos `/mlflow/artifacts`)
- `paths.py` — remap cross-container incl. `/mlflow/artifacts/`

## Pode importar

- `infra_ring`

## Não deve importar

- `platform_ring`, `executors_ring`, `orchestration_ring`

## Código actual (transição)

| Origem | Nota |
|--------|------|
| `src/core/ml/` | Shim de compatibilidade — importar de `ml_core_ring` |
| `train_backend.py` | Registry + backends registados em `domains/*/train_backend.py` |
| `orchestration_hooks.py` | `run_training_for_domain(domain, conf)` |
