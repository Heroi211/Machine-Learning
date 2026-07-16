# Roteiro de teste completo — Plataforma MLOps (TC01 + TC02)

Guia **passo a passo**: validação **manual** (UIs + Swagger) e, **no fim**, **gate automatizado** (scripts + pytest).

**Documentação geral:** [`DOCUMENTACAO.md`](DOCUMENTACAO.md)

**Tempo estimado:** 1ª vez ~60–90 min (manual + automático); repetição manual ~25 min + automático ~20 min.

---

## Escopo de testes

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  PARTE A — MANUAL (UIs + Swagger)                                           │
│  A0  Pré-requisitos + make docker-fresh                                    │
│  A1  Dozzle — contentores e logs                                            │
│  A2  pgAdmin — Postgres (3 bases + tabelas)                                 │
│  A3  Swagger — auth + health + rotas admin                                  │
│  A4  Swagger — TC02 recommendation (sync → promote → predict)             │
│  A5  Airflow UI — ml_training_dispatch                                    │
│  A6  MLflow UI — experiment + Registry tc02_recommender                     │
│  A7  DVC — pipeline offline (host)                                          │
│  A8  Swagger — TC01 churn (opcional)                                        │
│  A9  pgAdmin — conferência pós-testes (SQL)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  PARTE B — AUTOMATIZADO (gate final — terminal)                             │
│  B1  Infra script        validate_platform.sh --infra-only                  │
│  B2  Unitários           pytest platform_ring + ml_core_ring               │
│  B3  E2E plataforma      validate_platform.sh --skip-build                  │
│  B4  DVC metrics         dvc metrics show (opcional)                        │
│  B5  Smoke pytest         tests/smoke/ (opcional)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Tipo | O quê cobre | O quê **não** cobre |
|------|-------------|---------------------|
| **Manual** | Experiência real (Swagger, Airflow, MLflow, pgAdmin, Dozzle), DVC no host | Regressão repetível em CI |
| **Automatizado** | Infra, auth, worker, DAG, promote, Registry, predict, rollback, OpenAPI | Churn tabular (só manual Swagger) |

> **Ordem obrigatória:** concluir **Parte A** antes da **Parte B**. O script E2E (B3) cria runs e deployments — pode coexistir com o que fez no Swagger; o script faz dedupe de runs reco activos.

---

## Credenciais

| Serviço | Login | Senha |
|---------|-------|-------|
| API / Swagger / pgAdmin | `admin@admin.com` | `admin1` |
| Postgres (pgAdmin) | `admin` | `admin1` |
| Airflow | `airflow` | `airflow` |

| URL | Serviço |
|-----|---------|
| http://localhost:8000/docs | Swagger |
| http://localhost:8888 | Dozzle |
| http://localhost:5050 | pgAdmin |
| http://localhost:8080 | Airflow |
| http://localhost:5000 | MLflow |

---

# PARTE A — Testes manuais

## A0. Pré-requisitos e stack limpa

### A0.1 Software e `.env`

- Docker + Compose, Python 3.10+, portas livres (8000, 8080, 5000, 8010, 5432, 5050, 8888)
- `.env` com `ENVIRONMENT=development`, bases `processing` / `airflow` / `mlflow`, credenciais `admin`/`admin1`

```bash
cd /home/gabriel/Machine-Learning
cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
# Confirmar ENVIRONMENT=development no .env
```

### A0.2 Dados

- [ ] `data/recommendation/raw/ratings.csv`
- [ ] `params.yaml`, `dvc.yaml`
- [ ] (Churn) `ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### A0.3 Subir stack do zero

```bash
# Opcional: limpar artefactos MLflow antigos no host
rm -f src/artifacts/mlflow.db src/artifacts/mlruns/*.db
rm -rf src/artifacts/mlruns/*

make docker-fresh
```

Aguarde **3–5 min**. Confirme contentores Up (Dozzle ou `docker compose ps`).

```bash
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
```

- [ ] Stack Up  
- [ ] DAG `ml_training_dispatch` despausada  

---

## A1. Dozzle — contentores e logs

http://localhost:8888

- [ ] `database_processing`, `api_processing`, `worker_recommendation` — running
- [ ] `mlflow_server`, `airflow_webserver`, `airflow_scheduler` — running
- [ ] `pgadmin_db`, `dozzle_logs` — running
- [ ] Logs sem crash loop (API, worker, scheduler)

---

## A2. pgAdmin — Postgres (3 bases)

http://localhost:5050 → `admin@admin.com` / `admin1`

**Registar servidor** (1ª vez): host `database_processing`, port `5432`, user `admin`, pass `admin1`.

- [ ] Bases **`processing`**, **`airflow`**, **`mlflow`**
- [ ] `processing` → tabelas `users`, `pipeline_runs`, `deployed_models`, `predictions`
- [ ] Query: `SELECT email, role_id FROM users;` → `admin@admin.com`, role admin

---

## A3. Swagger — auth e shell da plataforma

http://localhost:8000/docs

### A3.1 Health (sem login)

Tag **`health`** → **`GET /v1/health`** → **200**, `"database": "up"`, `"environment": "development"`

### A3.2 Login e Authorize

1. **`POST /v1/auth/authenticate`** — `admin@admin.com` / `admin1` → copiar `access_token`
2. **Authorize** (cadeado) → `Bearer <token>`
3. **`GET /v1/auth/logged`** → **200**

### A3.3 Rotas admin (smoke)

| Endpoint | OK se |
|----------|--------|
| `GET /v1/users/` | 200 |
| `GET /v1/roles/` | 200 |
| `GET /v1/domains/recommendation/admin/runs` | 200 |
| `GET /v1/domains/churn/admin/runs` | 200 |

- [ ] Auth + rotas admin OK  

---

## A4. Swagger — TC02 Recommendation (fluxo principal)

Tag **`domain-recommendation`**

### A4.1 Treino sync (~2–5 min)

**`POST /v1/domains/recommendation/admin/train/sync`**

```json
{
  "train_models": ["torch_embedding"],
  "n_epochs": 1,
  "top_k": 5,
  "mlflow_experiment": "tc02_recommendation"
}
```

- [ ] **201** — `pipeline_run_id`, `mlflow_run_id`, `status: completed`, `champion_name`

### A4.2 Runs → Promote → Predict

| Passo | Endpoint | OK se |
|-------|----------|--------|
| Runs | `GET …/admin/runs?status=completed` | run recommendation completed |
| Promote | `POST …/admin/promote` | **201**, `mlflow_registry_model`: `tc02_recommender` |
| Predict | `POST …/predict` `{"user_id":1,"top_k":5}` | **200**, `recommended_items` com IDs |
| History | `GET …/admin/deployments/history` | deployment `active` |

### A4.3 Rollback (opcional)

Repetir sync + promote → **`POST …/admin/rollback`** → **200**

- [ ] TC02 Swagger E2E OK  

---

## A5. Airflow UI

http://localhost:8080 → `airflow` / `airflow`

**Via Swagger (disparo):** `POST /v1/domains/recommendation/admin/train/trigger` — form: `top_k=5`, `train_models=["torch_embedding"]`, `n_epochs=1`, `mlflow_experiment=tc02_recommendation` → **202**

- [ ] DAG **`ml_training_dispatch`** despausada
- [ ] Run recente → **success**
- [ ] (Após success) repetir promote + predict no Swagger se quiser validar pós-DAG

---

## A6. MLflow UI

http://localhost:5000

- [ ] Experiment **`tc02_recommendation`** — run com params, métricas, artefactos
- [ ] **Models** → **`tc02_recommender`** → alias **`Production`**

---

## A7. DVC — pipeline offline (host)

Instalação (se `dvc: not found`):

```bash
pip install "dvc>=3.0.0"
```

> Não é obrigatório `pip install -e .` — o editable do projecto pode falhar; basta DVC + deps já usadas pelo pipeline.

```bash
cd /home/gabriel/Machine-Learning
make tc02-repro
```

Conferir no IDE / explorador:

- [ ] `reports/recommendation/metrics.json` — champion + métricas @K
- [ ] `models/recommendation/torch_embedding.pt`
- [ ] `models/recommendation/champion_manifest.json`

Opcional: `dvc metrics show` · `dvc dag`

---

## A8. Swagger — TC01 Churn (opcional)

Tag **`domain-churn`**. Pré-requisito: `USE_MLP_FOR_PREDICTION=TRUE` no `.env`.

1. **`POST …/admin/train/baseline`** — form file = CSV Telco
2. **`POST …/admin/train/feature-engineering`** — mesmo CSV
3. **`POST …/admin/promote`**
4. **`POST …/predict`** — payload Telco (campos do schema `ChurnFeaturesInput`)

- [ ] Predict **200** com probabilidade / `inference_report`

---

## A9. pgAdmin — conferência final (manual)

Base **`processing`**, Query Tool:

```sql
SELECT id, pipeline_type, status, active FROM pipeline_runs ORDER BY id DESC LIMIT 5;
SELECT id, domain, status, pipeline_run_id FROM deployed_models ORDER BY id DESC;
SELECT id, domain, pipeline_run_id FROM predictions ORDER BY id DESC LIMIT 5;
```

- [ ] Runs, deployments e predictions coerentes com Swagger

---

### Checklist — Parte A concluída

- [ ] A1 Dozzle  
- [ ] A2 pgAdmin (3 bases)  
- [ ] A3 Swagger auth  
- [ ] A4 TC02 sync → promote → predict  
- [ ] A5 Airflow (opcional)  
- [ ] A6 MLflow Registry  
- [ ] A7 DVC repro  
- [ ] A8 Churn (opcional)  
- [ ] A9 SQL pgAdmin  

---

# PARTE B — Testes automatizados (gate final)

Executar **depois** da Parte A. Stack deve estar **Up**; DAG despausada.

## B1. Infraestrutura (~2 min)

Valida automaticamente: 8 contentores, health API/worker/Airflow/MLflow, Dozzle, pgAdmin, volumes reco, rede API→worker, DAG, Variable Airflow.

```bash
cd /home/gabriel/Machine-Learning
bash scripts/validate_platform.sh --skip-build --infra-only --skip-local
```

**OK se:** `FAIL: 0` (~20+ PASS)

---

## B2. Unitários — anéis (~30 s)

```bash
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
```

**OK se:** ~34 passed, 0 failed

Opcional:

```bash
PYTHONPATH=src:. python3 -m pytest tests/orchestration_ring/ tests/domains/ -q -o addopts=
```

> **Não usar** `make test-fast` como gate — inclui `tests/src/` (pandera × numpy 2.0).

Pré-requisitos locais (incluídos no B3):

```bash
python3 scripts/check_ring_imports.py
PYTHONPATH=src:. python3 scripts/validate_env.py
```

---

## B3. E2E plataforma (~15–20 min) — **gate principal**

```bash
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch

VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

Equivalente:

```bash
VALIDATE_API_PASSWORD=admin1 make validate-platform
```

(add `--skip-build` ao script directo; `make validate-platform` faz build se necessário)

### Fases validadas automaticamente

| Fase | Conteúdo |
|------|----------|
| 0 | `check_ring_imports` + `validate_env` |
| 1–2 | Docker, contentores, health, volumes, rede |
| 3 | Auth JWT, users, roles, runs |
| 4 | Worker `POST /train` + BD |
| 5 | API `train/trigger` → 202 |
| 6 | Airflow DAG → **success** |
| 7 | Promote, **MLflow Registry**, predict, rollback |
| 8 | OpenAPI `/domains/recommendation/` |

**OK se:**

```text
PASS: 46+   FAIL: 0   SKIP: 0
Validação da plataforma concluída com sucesso.
```

Variantes:

| Comando | Uso |
|---------|-----|
| `… --skip-build --skip-dag` | Sem esperar DAG (~5 min menos) |
| `… --skip-build --skip-rollback` | Sem teste rollback |
| `make validate-platform-infra` | Só infra |

- [ ] B1 infra 0 FAIL  
- [ ] B2 pytest verde  
- [ ] B3 E2E 0 FAIL  

---

## B4. DVC metrics (opcional, pós `make tc02-repro`)

```bash
dvc metrics show
```

- [ ] Métricas consistentes com `reports/recommendation/metrics.json`

---

## B5. Smoke pytest (opcional)

```bash
PYTHONPATH=src:. python3 -m pytest tests/smoke/ -q -o addopts=
```

---

# Checklist rubrica TC02

| # | Requisito | Parte |
|---|-----------|-------|
| R1 | Pipeline DVC | A7 (+ B4 opcional) |
| R2 | PyTorch embedding | A4 / A7 |
| R3 | Baselines (params.yaml) | A7 |
| R4 | MLflow tracking | A6 |
| R5 | Model Registry Production | A6 + B3 |
| R6 | API JWT | A3 |
| R7 | Airflow dispatch | A5 + B3 |
| R8 | Promote | A4 + B3 |
| R9 | Predict online | A4 + B3 |
| R10 | Rollback | A4 + B3 |
| R11 | Métricas @K | A7 |
| R12 | Docker compose | A0 + B1 |

---

# Troubleshooting

| Sintoma | Acção |
|---------|--------|
| `dvc: not found` | `pip install "dvc>=3.0.0"` |
| `pip install -e .` falha | Usar `pip install dvc`; pytest com `PYTHONPATH=src:.` |
| Train sync 502 | Dozzle → logs `worker_recommendation` |
| Promote 400 ambiguidade | Correr B3 (dedupe automático) |
| DAG queued | `docker exec airflow_scheduler airflow dags unpause ml_training_dispatch` |
| B3 FAIL fase 6 | Airflow UI → logs da task; confirmar DAG despausada |

SQL dedupe reco (pgAdmin): ver [`DOCUMENTACAO.md`](DOCUMENTACAO.md) §12.

---

# Sessão completa — copy-paste

```bash
cd /home/gabriel/Machine-Learning

# ── PARTE A0 ──
cp .env_example .env && echo "AIRFLOW_UID=$(id -u)" >> .env
make docker-fresh
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch

# ── PARTE A7 (DVC) ──
pip install "dvc>=3.0.0"
make tc02-repro

# ── PARTE A (manual) ──
# Dozzle :8888 → pgAdmin :5050 → Swagger :8000/docs
# TC02: sync → promote → predict → MLflow :5000 → Airflow :8080

# ── PARTE B (automático) ──
bash scripts/validate_platform.sh --skip-build --infra-only --skip-local
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

---

*Última actualização: escopo manual (Parte A) + gate automatizado (Parte B) no fim.*
