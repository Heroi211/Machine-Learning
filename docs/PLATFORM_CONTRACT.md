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

Prefixo global: `{PROJECT_VERSION}` (ex.: `/v1`).

**Contrato preferido (Fase 5.5):** domínio **no path**, não no body nem em `?domain=`.

Prefixo por domínio: **`/v1/domains/{domain}/`**

| Método | Rota (relativa ao domínio) | Auth | Propósito |
|--------|----------------------------|------|-----------|
| POST | `/predict` | User | Inferência (schema por domínio, sem campo `domain`) |
| GET | `/admin/runs` | Admin | Listar runs do domínio |
| POST | `/admin/promote` | Admin | Activar modelo para `/predict` |
| POST | `/admin/rollback` | Admin | Reverter deployment |
| GET | `/admin/deployments/history` | Admin | Histórico |
| POST | `/admin/train/trigger` | Admin | Dispara treino no Airflow |
| POST | `/admin/train/sync` | Admin | Treino sync (reco, debug) |
| POST | `/admin/train/baseline` | Admin | Baseline sync (churn, debug) |
| POST | `/admin/train/feature-engineering` | Admin | FE sync (churn, debug) |

Rotas globais (fora do domínio):

| Método | Rota | Auth | Propósito |
|--------|------|------|-----------|
| POST | `/auth/authenticate` | — | Login → JWT |
| GET | `/health` | — | Liveness |

**Domínios actuais:** `churn`, `recommendation`.

**Extensão permitida:** novo valor de `{domain}` + schema Pydantic de `/predict` — **não** novas rotas de ciclo de vida por problema.

### Legado (deprecado — apagar após migração frontend)

| Método | Rota legada | Substituir por |
|--------|-------------|----------------|
| POST | `/processor/admin/train/trigger-dag` | `/domains/{domain}/admin/train/trigger` |
| GET | `/processor/admin/runs?domain=` | `/domains/{domain}/admin/runs` |
| POST | `/processor/admin/promote?domain=` | `/domains/{domain}/admin/promote` |
| POST | `/processor/predict` (+ `domain` no body) | `/domains/{domain}/predict` |

Ver `docs/DECISOES_REBUILD_DOMINIOS.md` e `MAP.md`.

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
5. **Router** em `platform_ring/domains/<nome>/router.py` — rotas `/v1/domains/<nome>/…`.
6. **Schema** em `schemas/` — entrada de `/predict` (sem campo `domain`).

**Não obrigatório alterar:** auth, promote, rollback, estrutura de `pipeline_runs`, padrão train/trigger.

---

## Treino: Airflow como orquestrador central

- **Churn / tabular:** DAG dispatch (branch tabular) ou sync debug em dev.
- **Recomendação / DVC:** worker HTTP — orquestrada pelo Airflow.
- **API manual síncrona** (baseline/FE/sync reco em dev): excepção de desenvolvimento; produção = Airflow.

---

## Referências

- [MAP.md](../MAP.md) — mapa rápido de rotas e paths.
- [DECISOES_REBUILD_DOMINIOS.md](./DECISOES_REBUILD_DOMINIOS.md) — decisões Fase 5.5.
- [ARCHITECTURE_RINGS.md](./ARCHITECTURE_RINGS.md) — anéis, pastas, diagramas.
- [IMPORT_RULES_RINGS.md](./IMPORT_RULES_RINGS.md) — dependências entre módulos.
- [MIGRATION_RINGS.md](./MIGRATION_RINGS.md) — fases de migração.
