# Regras de import entre anéis (`_ring`)

Fase 0 — matriz de dependências. Violações devem falhar em `scripts/check_ring_imports.py`.

---

## Anéis (do mais externo ao mais interno)

| Anel | Pasta | Pode importar |
|------|-------|---------------|
| 0 | `infra_ring` | stdlib, libs externas |
| 2 | `ml_core_ring` | `infra_ring` |
| 2b | `domains_ring` | `ml_core_ring`, `infra_ring` |
| 3 | `executors_ring` | `ml_core_ring`, `domains_ring`, `infra_ring` |
| 4 | `orchestration_ring` | `ml_core_ring`, `executors_ring`, `infra_ring` |
| 1 | `platform_ring` | `ml_core_ring`, `infra_ring` |

---

## Proibições explícitas

| Origem | **Não** importar |
|--------|------------------|
| `platform_ring` | `executors_ring`, `orchestration_ring` |
| `ml_core_ring` | `platform_ring`, `executors_ring`, `orchestration_ring` |
| `domains_ring` | `platform_ring`, `executors_ring`, `orchestration_ring` |
| `executors_ring` | `platform_ring`, `orchestration_ring` |
| `orchestration_ring` | `platform_ring` |

**Motivo:** a plataforma delega via **registries** (`get_train_backend`, `get_engine`), nunca via imports directos de implementação.

---

## Código legado (transição)

Até concluir a migração, o mapa abaixo aplica-se por **intenção**:

| Pasta actual | Anel alvo |
|--------------|-----------|
| `src/api/`, `src/services/processor/`, `src/services/auth/` | `platform_ring` |
| `src/core/ml/`, `src/domains/` | `ml_core_ring` + `domains_ring` |
| `src/services/pipelines/` | `executors_ring/tabular_classification` |
| `src/domains/recommendation/` (runner, models) | `executors_ring/recommendation` |
| `airflow/dags/` | `orchestration_ring/airflow/dags` |
| `src/core/configs.py`, `database.py` | `infra_ring` |

O script de verificação inclui regras **transitórias** para pastas legadas (ver comentários no script).

---

## Como verificar

```bash
python3 scripts/check_ring_imports.py
```

Integrar em CI / pre-commit na Fase 1.
