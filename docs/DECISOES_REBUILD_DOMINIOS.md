# Decisões — rebuild Opção A + rotas sync

Registo fechado em **2026-06-25** antes da implementação das rotas `/v1/domains/{domain}/…`.

---

## Decisões confirmadas

| # | Tema | Decisão |
|---|------|---------|
| **1** | Reco sync (treino manual sem Airflow) | **Sim** — `POST …/domains/recommendation/admin/train/sync` |
| **2** | Quem pode treinar (sync + trigger API) | **Já existe** — `ENVIRONMENT=prd` bloqueia; `development` + **admin** (`require_sync_training_routes_enabled`, `require_airflow_api_trigger_enabled` em `src/core/deps.py`). **Não** criar flag extra. |
| **3** | Predict | **Uma rota por domínio** — sem `domain` no body; sem union/`if` central |
| **4** | `OBJECTIVE` / `?domain=` | **Eliminar** nas rotas novas; domínio **só no path** |
| **5** | Rotas legadas `/v1/processor/…` | **Manter por agora** → apagar depois dos testes (ver lembretes) |
| **6** | DAG `ml_training_pipeline` | **Manter por agora** → apagar depois do tabular validado na nova stack |
| **7** | Debug | **Docker** (BD, worker, Airflow, MLflow) + **API local** (`uvicorn` / `make run`) com breakpoints |
| **8** | TC03 | **Depois** — pastas por domínio (Opção A) pavimentam; sem registry genérico agora |

---

## MLflow vs deployed_models (Fase 6)

| Camada | Responsabilidade |
|--------|------------------|
| **`deployed_models` (Postgres)** | Fonte de verdade do `/predict`, rollback API, 1 activo por domain |
| **MLflow Tracking** | Runs, métricas, artefactos no treino |
| **MLflow Registry** | Side-effect no `POST …/admin/promote` — Staging → Production |

Nomes Registry: `tc02_recommender` (reco), `churn_fe_model` (churn). Falha Registry → aviso em `mlflow_registry_warning`; promote na BD **não** falha.

---

## Pacote de rotas por domínio (alvo)

Prefixo: `/v1/domains/{domain}/`

### Automático (Airflow — admin, `ENVIRONMENT≠prd` para trigger API)

| Método | Rota | Função |
|--------|------|--------|
| POST | `…/admin/train/trigger` | Dispara `ml_training_dispatch` |

### Manual sync (admin, `ENVIRONMENT≠prd`)

| Domínio | Rotas |
|---------|--------|
| **churn** | `…/admin/train/baseline`, `…/admin/train/feature-engineering` |
| **recommendation** | `…/admin/train/sync` |

### Operação manual (admin / user)

| Método | Rota | Auth |
|--------|------|------|
| GET | `…/admin/runs` | Admin |
| POST | `…/admin/promote` | Admin |
| GET | `…/admin/deployments/history` | Admin |
| POST | `…/admin/rollback` | Admin |
| POST | `…/predict` | User |

---

## Global (não muda)

- `/v1/auth/*`, `/v1/users/*`, `/v1/roles/*`, `/v1/health`
- Tabelas: `pipeline_runs`, `deployed_models`, `predictions`
- DAG canónica de treino: **`ml_training_dispatch`**

---

## Debug local (decisão 7)

Stack Docker sobe serviços de apoio; **API corre no host**:

```bash
# Terminais típicos
docker compose up -d db_processing worker_recommendation mlflow_server airflow-scheduler airflow-webserver
make run   # uvicorn :8000, PYTHONPATH=src
```

- Breakpoints em `src/platform_ring/domains/…` e `services/processor/…` no processo local.
- `.env`: `DATABASE_SERVER=localhost`, paths `src/…` no host; worker/reco sync ainda em `localhost:8010` se usar rota sync reco.
- Validar que credenciais BD e URLs batem com compose (porta 5432 exposta).

---

## Lembretes pós-implementação (TODO)

**Fase 5.5 concluída (2026-06-25):** rotas `/v1/domains/{churn,recommendation}/…` activas; legado mantido.

**Fase 6 concluída:** promote na API espelha MLflow Registry (`tc02_recommender`, `churn_fe_model`); BD `deployed_models` mantém-se como runtime de `/predict`. Side-effect **best-effort** — falha Registry não reverte promote na BD.

Apagar **depois** de churn + reco passarem nas rotas novas + testes manuais sync:

- [ ] Rotas legadas em `src/api/v1/endpoints/processor.py` (`/processor/admin/runs`, `/promote`, `/predict` union, etc.)
- [ ] DAG `airflow/dags/ml_training_pipeline.py`
- [ ] Variable/bootstrap só legado se deixar de ser usada (`ml_training_pipeline_conf`)
- [ ] Uso de `settings.objective` como default de API (manter só se algum script interno precisar)
- [ ] Actualizar `scripts/validate_platform.sh` para rotas `/domains/…`
- [ ] Actualizar `docs/PLATFORM_CONTRACT.md` (rotas por domínio)

---

## Estrutura de código (alvo)

```text
src/platform_ring/
  domains/
    churn/
      router.py
      predict.py
      promote.py
      runs.py
      train_trigger.py
      train_sync.py      # baseline + FE
    recommendation/
      router.py
      predict.py
      …
      train_sync.py      # proxy worker
```

Lógica ML pesada continua em `executors_ring/` e `services/pipelines/` (tabular) até Fase 7b; rotas novas **delegam**, não duplicam algoritmos.

---

## Referências

- Análise arquitectural: `docs/ARQUITETURA_PLATAFORMA.md`
- Controlo ENV: `src/core/deps.py`, `settings.is_production` em `src/core/configs.py`
