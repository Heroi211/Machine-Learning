# Plano A — Core ML modular

> **Arquitectura alvo:** ver [ARCHITECTURE_RINGS.md](./ARCHITECTURE_RINGS.md) e [PLATFORM_CONTRACT.md](./PLATFORM_CONTRACT.md).

Camada compartilhada em `src/core/ml/` (migrará para `src/ml_core_ring/`) para novos domínios e motores sem reescrever Fase 01.

## Componentes

| Módulo | Função |
|--------|--------|
| `artifact_manifest.py` | Contrato de artefactos promovidos |
| `inference_engine.py` | Registry de motores (`sklearn_joblib`, `torch_bundle`) |
| `domain_plugin.py` | Registry único de domínios |
| `pipeline_runner.py` | Template Method para pipelines DVC |
| `engines/` | Adapters sobre código legado |

## Domínios

- `domains/churn/` — Fase 01 (legado)
- `domains/recommendation/` — TC02 (DVC + embedding PyTorch)

## Comandos TC02

```bash
# Pipeline reprodutível
PYTHONPATH=src dvc repro

# MLflow Registry
PYTHONPATH=src python scripts/ml/promote_registry.py

# Docker overlay
docker compose -f docker-compose.tc02.yml up --build
```

## Fase 01

Inalterada: `docker compose up`, Airflow, `/predict` churn via `InferenceEngine`.
