# Contrato imutável da plataforma ML

Documento de **Fase 0** — define o fluxo de produto que **não muda** quando surgir um novo tipo de problema ML (TC03, TC04…).

---

## Princípio

> Novos domínios **plugam** na plataforma. A plataforma **não redesenha** login, ciclo de vida de modelos nem inferência.

---

## Fluxo canónico (4 passos)

```text
1. TREINAR     → orquestration_ring (Airflow) dispara executor via registry
2. REGISTRAR   → platform_ring persiste pipeline_runs (status, métricas, artefactos)
3. PROMOVER    → platform_ring POST promote → deployed_models (1 activo por domain)
4. PREDIZER    → platform_ring POST /predict → predictions + InferenceEngine
```

Autenticação (**JWT + roles**) envolve todos os passos expostos na API.

---

## Responsabilidades por anel

| Passo | Anel responsável | Não responsável |
|-------|------------------|-----------------|
| Login / users | `platform_ring` | executors |
| Disparo de treino | `orchestration_ring` (Airflow) | lógica de algoritmo na API |
| Execução ML pesada | `executors_ring` | HTTP público |
| Registo de runs | `platform_ring` + BD | — |
| Promote / rollback | `platform_ring` | MLflow Registry (side-effect) |
| Predict | `platform_ring` + `ml_core_ring` (engines) | treino |

---

## Rotas HTTP estáveis (contrato frontend)

Prefixo: `{PROJECT_VERSION}` (ex.: `/v1`).

| Método | Rota | Auth | Propósito |
|--------|------|------|-----------|
| POST | `/auth/authenticate` | — | Login → JWT |
| POST | `/processor/admin/train/trigger-dag` | Admin | Dispara treino no Airflow (`domain` no conf) |
| GET | `/processor/admin/runs` | Admin | Listar runs (`domain`, `status`, …) |
| POST | `/processor/admin/promote` | Admin | Activar modelo para `/predict` |
| POST | `/processor/admin/rollback` | Admin | Reverter deployment |
| GET | `/processor/admin/deployments/{domain}/history` | Admin | Histórico |
| POST | `/processor/predict` | User | Inferência (`domain` + payload por schema) |
| GET | `/health` | — | Liveness |

**Extensão permitida:** novos valores de `domain` e schemas Pydantic — **não** novas rotas de ciclo de vida por problema.

---

## Modelo de dados (imutável)

| Tabela | Papel |
|--------|--------|
| `users` / `roles` | Identidade |
| `pipeline_runs` | Todo treino (qualquer `domain`) |
| `deployed_models` | Modelo activo por `domain` |
| `predictions` | Auditoria de inferência |

Campos novos podem ser adicionados em `metrics` (JSON) ou colunas opcionais — **não** tabelas paralelas por domínio.

---

## O que cada novo problema DEVE fornecer

1. **`domains_ring/<nome>/plugin.py`** — `DomainPlugin` registado.
2. **`executors_ring/<tipo>/`** — implementação de treino (`TrainBackend`).
3. **`ml_core_ring`** — registo em `TRAIN_BACKEND_REGISTRY` e, se servir online, `InferenceEngine`.
4. **`orchestration_ring`** — task Airflow (ou branch na DAG dispatch) que invoca o backend.
5. **Schema** em `platform_ring/schemas/` — entrada de `/predict` (union por `domain`).

**Não obrigatório alterar:** auth, promote, rollback, estrutura de `pipeline_runs`, padrão trigger-dag.

---

## Treino: Airflow como orquestrador central

- **Churn / tabular:** DAG existente ou branch `legacy_fe` na DAG dispatch.
- **Recomendação / DVC:** task `dvc repro` ou worker HTTP — orquestrada pelo Airflow.
- **API manual síncrona** (baseline/FE em dev): excepção de desenvolvimento; produção = Airflow.

---

## Referências

- [ARCHITECTURE_RINGS.md](./ARCHITECTURE_RINGS.md) — anéis, pastas, diagramas.
- [IMPORT_RULES_RINGS.md](./IMPORT_RULES_RINGS.md) — dependências entre módulos.
- [MIGRATION_RINGS.md](./MIGRATION_RINGS.md) — fases de migração.
