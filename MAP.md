# Mapa do repositório — onde começar

## Global (todos os domínios)

| O quê | Onde |
|-------|------|
| Login JWT | `POST /v1/auth/authenticate` |
| Users / roles | `/v1/users`, `/v1/roles` |
| Health | `/v1/health` |

## Rotas por domínio (Fase 5.5 — preferir estas)

Prefixo: **`/v1/domains/{domain}/`**

| Operação | churn | recommendation |
|----------|-------|----------------|
| Predict | `POST …/churn/predict` | `POST …/recommendation/predict` |
| Runs | `GET …/churn/admin/runs` | `GET …/recommendation/admin/runs` |
| Promote | `POST …/admin/promote` | idem |
| Deploy history | `GET …/admin/deployments/history` | idem |
| Rollback | `POST …/admin/rollback` | idem |
| Treino auto (Airflow) | `POST …/admin/train/trigger` + CSV | `POST …/admin/train/trigger` |
| Treino sync (debug) | `…/train/baseline`, `…/train/feature-engineering` | `…/admin/train/sync` |

Swagger: tags **`domain-churn`** e **`domain-recommendation`**.

## Legado (apagar depois — ver `docs/DECISOES_REBUILD_DOMINIOS.md`)

- `/v1/processor/*` — tag **`processor (legacy)`**
- DAG `ml_training_pipeline`

## Código por domínio

| TC | Rotas HTTP | ML / treino | Dados |
|----|------------|-------------|-------|
| **Churn** | `src/platform_ring/domains/churn/` | `services/pipelines/`, `executors_ring/tabular_classification/` | `ml_data/uploads/`, `src/data/`, `src/artifacts/models/` |
| **Reco** | `src/platform_ring/domains/recommendation/` | `domains/recommendation/`, worker `:8010` | `data/recommendation/`, `models/recommendation/` |

## Paths (host local, `ML_PROJECT_ROOT` = raiz do repo)

| Variável / helper | Valor |
|-------------------|--------|
| Uploads CSV | `ml_data/uploads/` (`ML_SHARED_PATH`) |
| Conf Airflow CSV | `/opt/airflow/ml_project/uploads/{ficheiro}` (via `airflow_upload_path()`) |
| Volume Docker API/Airflow | `ml_shared` montado em `/var/www/ml_shared` / `/opt/airflow/ml_project` |
| Artefactos reco | `models/recommendation/` |
| Remap BD → local | `ml_core_ring.paths.resolve_shared_artifact_path` |

**Env:** usar `ML_SHARED_PATH=ml_data/uploads` (não `ml_shared/uploads`). Ficheiros espelho: `.env_example` e `.env.example` — manter iguais.

## Debug local (API uvicorn + Docker serviços)

```bash
docker compose up -d db_processing worker_recommendation mlflow_server airflow-scheduler airflow-webserver
# .env: ENVIRONMENT=development, DATABASE_SERVER=localhost, WORKER_RECOMMENDATION_URL=http://localhost:8010
make run
```

## Documentação

- Decisões rebuild: `docs/DECISOES_REBUILD_DOMINIOS.md`
- Arquitectura: `docs/ARQUITETURA_PLATAFORMA.md`
- Contrato produto: `docs/PLATFORM_CONTRACT.md`
