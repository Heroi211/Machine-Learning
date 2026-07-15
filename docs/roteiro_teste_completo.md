# Roteiro de teste completo — TC02 (Recommendation)

Guia **passo a passo** para subir a stack do zero, validar o domínio **recommendation** (Tech Challenge 02), conferir **DVC**, **MLflow** (tracking + **Model Registry**) e testar via **Swagger**.

**Documentação geral:** [`DOCUMENTACAO.md`](DOCUMENTACAO.md)

**Tempo estimado:** 1ª execução ~45–60 min (`docker-fresh` + DVC + E2E); repetição ~20 min.

---

## Mapa do roteiro

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  0. Pré-requisitos host     Python, Docker, dados MovieLens              │
│  1. Ambiente limpo          make docker-fresh                            │
│  2. Smoke infra             health, Postgres 3 bases, OpenAPI            │
│  3. DVC (offline)           dvc repro → métricas + artefactos            │
│  4. Swagger — auth          JWT admin                                    │
│  5. Swagger — reco E2E      sync → runs → promote → predict → rollback   │
│  6. Airflow (opcional)      ml_training_dispatch via trigger             │
│  7. MLflow UI               experiments + Registry tc02_recommender      │
│  8. Script automático       validate_platform.sh                         │
│  9. Checklist TC02          rubrica fechada                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 0. Pré-requisitos

### 0.1 Software

| Item | Versão mínima |
|------|----------------|
| Docker + Compose | 24 / 2.20 |
| Python | 3.10+ |
| Git | qualquer recente |

Portas livres: **8000** (API), **8080** (Airflow), **5000** (MLflow), **8010** (worker), **5432**, **5050**, **8888**.

### 0.2 Repositório e `.env`

```bash
cd /home/gabriel/Machine-Learning

cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
```

Ajuste no `.env` (obrigatório para testes):

| Variável | Valor para testes |
|----------|-------------------|
| `ENVIRONMENT` | **`development`** |
| `DATABASE_USER` / `DATABASE_PASS` | `admin` / `admin1` |
| `DATABASE_NAME` | `processing` |
| `AIRFLOW_DATABASE_NAME` | `airflow` |
| `MLFLOW_DATABASE_NAME` | `mlflow` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `MLFLOW_ARTIFACT_ROOT` | `src/artifacts/mlruns` |
| `ML_SHARED_PATH` | `ml_data/uploads` |

> No Docker Compose a API/worker usam `MLFLOW_TRACKING_URI=http://mlflow_server:5000` e `WORKER_RECOMMENDATION_URL=http://worker_recommendation:8010` (já definidos no compose).

### 0.3 Dados TC02 (MovieLens)

Confirme antes de subir a stack:

```bash
ls -la data/recommendation/raw/ratings.csv
ls -la params.yaml
ls -la dvc.yaml
```

**OK se:** `ratings.csv` existe (~2,4 MB). O zip `ml-latest-small.zip` na mesma pasta é opcional (já extraído).

Se faltar `ratings.csv`:

```bash
# Exemplo: extrair do zip incluído no repo
unzip -o data/recommendation/raw/ml-latest-small.zip -d /tmp/ml-small
cp /tmp/ml-small/ratings.csv data/recommendation/raw/ratings.csv
```

### 0.4 Dependências Python (host — DVC e pytest)

```bash
make install-dev
# ou, com extra TC02 (inclui dvc):
pip install -e ".[dev,tc02]"
```

Validação rápida:

```bash
python3 scripts/validate_env.py
# Esperado: Ambiente OK — core ML + TC02 presentes.
```

---

## 1. Subir stack do zero (`make docker-fresh`)

Reset **total**: volumes Postgres (3 bases), metadados Airflow, rebuild sem cache.

```bash
cd /home/gabriel/Machine-Learning

# Opcional: limpar runs MLflow antigos no disco
rm -f src/artifacts/mlflow.db src/artifacts/mlruns/*.db
rm -rf src/artifacts/mlruns/*

make docker-fresh
```

> Preserva no host: `.env`, `data/recommendation/`, `models/recommendation/`, `ml_data/uploads/`.

Aguarde **3–5 minutos** e confirme:

```bash
docker compose ps
```

**OK se** todos relevantes estão `Up` (ou `healthy`):

| Contentor | Papel |
|-----------|--------|
| `database_processing` | Postgres |
| `api_processing` | API :8000 |
| `worker_recommendation` | Worker reco :8010 |
| `mlflow_server` | MLflow :5000 |
| `airflow_webserver` / `airflow_scheduler` | Airflow :8080 |
| `pgadmin_db` | pgAdmin :5050 |

Despausar DAG canónica:

```bash
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch
```

- [ ] Stack Up  
- [ ] DAG `ml_training_dispatch` despausada  

---

## 2. Smoke — infraestrutura

### 2.1 Health checks

```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
curl -s http://localhost:8010/health | python3 -m json.tool
curl -s http://localhost:8080/health | python3 -m json.tool
curl -s http://localhost:5000/api/2.0/mlflow/experiments/search \
  -H "Content-Type: application/json" \
  -d '{"max_results":1}' | python3 -m json.tool
```

**OK se:**

- API → `"environment": "development"`
- Worker → status OK
- Airflow → healthy
- MLflow → JSON com experiments (pode estar vazio na 1ª vez)

### 2.2 OpenAPI — rotas recommendation

```bash
curl -s http://localhost:8000/openapi.json \
  | python3 -c "
import json,sys
paths=json.load(sys.stdin)['paths']
reco=[k for k in paths if '/domains/recommendation/' in k]
print(f'Rotas recommendation: {len(reco)}')
for p in sorted(reco):
    print(' ', p)
"
```

**OK se:** ≥ 8 rotas, incluindo:

- `POST /v1/domains/recommendation/predict`
- `POST /v1/domains/recommendation/admin/promote`
- `POST /v1/domains/recommendation/admin/train/sync`
- `POST /v1/domains/recommendation/admin/train/trigger`
- `GET /v1/domains/recommendation/admin/runs`

### 2.3 Postgres — 3 bases

```bash
docker exec database_processing psql -U admin -d processing -c "\l" \
  | grep -E 'processing|airflow|mlflow'
```

- [ ] Bases `processing`, `airflow`, `mlflow` existem  

### 2.4 Volumes reco no worker

```bash
docker exec worker_recommendation test -f /opt/ml_project/data/recommendation/ratings.csv \
  && echo "ratings.csv OK no worker"
docker exec api_processing test -f /opt/ml_project/params.yaml \
  && echo "params.yaml OK na API"
```

- [ ] Smoke infra OK  

---

## 3. DVC — pipeline offline (TC02)

O **DVC** valida o pipeline ML **fora do Docker**: preprocess → feature_eng → train → evaluate.

Requisito TC02: pipeline reprodutível com `dvc.yaml` + `params.yaml` + métricas em `reports/`.

### 3.1 Instalar DVC (se ainda não tiver)

```bash
pip install "dvc>=3.0.0"
# ou: pip install -e ".[dev,tc02]"
```

### 3.2 Executar pipeline completo

```bash
cd /home/gabriel/Machine-Learning
make tc02-repro
# equivalente: PYTHONPATH=src dvc repro
```

**OK se** termina sem erro e gera/atualiza:

| Artefacto | Caminho |
|-----------|---------|
| Interações processadas | `data/recommendation/processed/interactions.parquet` |
| Features treino | `data/recommendation/features/train.parquet` |
| Modelo campeão | `models/recommendation/torch_embedding.pt` |
| Métricas treino | `models/recommendation/train_metrics.json` |
| Métricas avaliação | `reports/recommendation/metrics.json` |
| Manifest campeão | `models/recommendation/champion_manifest.json` |

### 3.3 Conferir métricas DVC

```bash
cat reports/recommendation/metrics.json | python3 -m json.tool
cat models/recommendation/champion_manifest.json | python3 -m json.tool
```

**OK se:**

- `champion` = `torch_embedding` (ou modelo definido em `params.yaml`)
- Métricas presentes: `hit_rate`, `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `map_at_k` (conforme implementação)

### 3.4 DVC metrics (opcional)

```bash
dvc metrics show
dvc dag
```

- [ ] `dvc repro` verde  
- [ ] `reports/recommendation/metrics.json` actualizado  

> **Nota:** O runtime da API usa treino via **worker** (sync ou Airflow), não `dvc repro` directamente. O DVC prova reprodutibilidade do pipeline TC02; a plataforma prova integração MLOps.

---

## 4. Swagger — autenticação

Abrir: **http://localhost:8000/docs**

### 4.1 Login

1. Expandir **`POST /v1/auth/authenticate`**
2. **Try it out**
3. Preencher:
   - `username`: `admin@admin.com`
   - `password`: `admin1`
4. **Execute**

**OK se:** resposta **200** com `access_token`.

### 4.2 Authorize global

1. Clicar **Authorize** (cadeado no topo)
2. Colar: `Bearer <access_token>` (ou só o token, conforme UI)
3. **Authorize** → **Close**

### 4.3 Confirmar sessão

- **`GET /v1/auth/logged`** → 200, utilizador admin

Credenciais de referência:

| Serviço | Login | Senha |
|---------|-------|-------|
| API / Swagger | `admin@admin.com` | `admin1` |
| Airflow | `airflow` | `airflow` |
| pgAdmin | `admin@admin.com` | `admin1` |
| Postgres | `admin` | `admin1` |

- [ ] Token JWT obtido e Authorize activo  

---

## 5. Swagger — fluxo recommendation (E2E manual)

Tag: **`domain-recommendation`**

Ordem recomendada na **1ª sessão** após `docker-fresh`:

### 5.1 Treino sync (rápido — ~2–5 min)

**`POST /v1/domains/recommendation/admin/train/sync`**

Request body (JSON):

```json
{
  "train_models": ["torch_embedding"],
  "n_epochs": 1,
  "top_k": 5,
  "mlflow_experiment": "tc02_recommendation"
}
```

**OK se:** **201** com:

- `pipeline_run_id` (inteiro)
- `mlflow_run_id` (string UUID)
- `status`: `"completed"`
- `champion_name`: `"torch_embedding"` (ou modelo vencedor)
- `metrics` com valores numéricos

> Se falhar com erro de worker: `docker compose logs worker_recommendation --tail 50`

### 5.2 Listar runs

**`GET /v1/domains/recommendation/admin/runs`**

Query opcional: `status=completed`

**OK se:** lista contém o run do passo 5.1 com `pipeline_type=recommendation`, `status=completed`.

### 5.3 Promote (activar modelo para `/predict`)

**`POST /v1/domains/recommendation/admin/promote`**

Sem body.

**OK se:** **201** com:

| Campo | Esperado |
|-------|----------|
| `status` | `active` |
| `domain` | `recommendation` |
| `pipeline_run_id` | id do run promovido |
| `mlflow_registry_model` | `tc02_recommender` |
| `mlflow_registry_version` | número da versão (ideal) |
| `mlflow_registry_stage` | `Production` |

Se `mlflow_registry_warning` aparecer mas HTTP 201: promote na **BD OK**; rever MLflow no passo 7.

### 5.4 Predict

**`POST /v1/domains/recommendation/predict`**

Request body:

```json
{
  "user_id": 1,
  "top_k": 5
}
```

**OK se:** **200** com:

- `recommended_items`: array com até 5 inteiros (item_id)
- `domain`: `"recommendation"`
- `pipeline_run_id`: coincide com deployment activo

Exemplo de resposta válida:

```json
{
  "recommended_items": [1272, 1209, 1250, 3741, 246],
  "domain": "recommendation",
  "prediction": 1272
}
```

### 5.5 Histórico de deployments

**`GET /v1/domains/recommendation/admin/deployments/history`**

**OK se:** **200**, lista com pelo menos 1 entrada `active`.

### 5.6 Rollback (opcional — precisa de 2 promotes)

1. Repetir **5.1** (segundo treino) + **5.3** (segundo promote)
2. **`POST /v1/domains/recommendation/admin/rollback`**

**OK se:** **200**, deployment anterior reactivado.

- [ ] Train sync OK  
- [ ] Promote OK  
- [ ] Predict OK com `recommended_items`  
- [ ] (Opcional) Rollback OK  

---

## 6. Airflow — treino via orquestração (opcional)

Valida integração **API → Airflow → worker** (mesmo caminho do script automático).

### 6.1 Swagger trigger

**`POST /v1/domains/recommendation/admin/train/trigger`**

Form fields:

| Campo | Valor teste |
|-------|-------------|
| `top_k` | `5` |
| `train_models` | `["torch_embedding"]` |
| `n_epochs` | `1` |
| `mlflow_experiment` | `tc02_recommendation` |

**OK se:** **202** com `dag_run_id` e `dag_id`: `ml_training_dispatch`.

### 6.2 Acompanhar na UI

1. http://localhost:8080 → login `airflow` / `airflow`
2. DAG **`ml_training_dispatch`** → run recente → **success**

### 6.3 Após success

Repetir no Swagger: **runs → promote → predict** (secção 5.2–5.4).

- [ ] DAG success no Airflow  

---

## 7. MLflow — tracking e Model Registry

UI: **http://localhost:5000**

### 7.1 Experiment

1. Menu **Experiments**
2. Procurar experimento **`tc02_recommendation`**
3. Abrir run mais recente (do sync ou DAG)

**OK se:** run contém:

- Parâmetros: `n_epochs`, `top_k`, `train_models`, etc.
- Métricas: hit rate, NDCG, etc.
- Artefactos: modelo PyTorch / bundle (subpasta `pytorch_model` ou similar)

### 7.2 Model Registry

1. Menu **Models**
2. Modelo registrado: **`tc02_recommender`**
3. Versão promovida com alias **`Production`**

Confirmação via REST:

```bash
curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/get?name=tc02_recommender" \
  | python3 -m json.tool
```

Ou versões:

```bash
curl -s "http://localhost:5000/api/2.0/mlflow/model-versions/search" \
  -H "Content-Type: application/json" \
  -d '{"filter": "name='"'"'tc02_recommender'"'"'"}' \
  | python3 -m json.tool
```

**OK se:** pelo menos 1 versão; alias Production aponta para versão do promote.

### 7.3 CLI promote Registry (opcional)

Se promote via API não preencheu Registry:

```bash
# Usar mlflow_run_id do passo 5.1
MLFLOW_RUN_ID=<uuid-do-run> make tc02-promote
# ou:
PYTHONPATH=src python3 scripts/ml/promote_registry.py --domain recommendation --mlflow-run-id <uuid>
```

### 7.4 Artefactos no host

```bash
ls -la src/artifacts/mlruns/ | head
```

Metadados MLflow vivem em Postgres (`mlflow`); ficheiros em `src/artifacts/mlruns/`.

- [ ] Experiment `tc02_recommendation` visível  
- [ ] Model `tc02_recommender` com versão Production  

---

## 8. Validação automática (gate oficial)

Com stack **já Up** e DAG despausada:

```bash
cd /home/gabriel/Machine-Learning

PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=

VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

**Resultado esperado:**

```text
PASS: 46+   FAIL: 0   SKIP: 0
```

Inclui: worker `/train`, trigger Airflow, DAG success, promote, **MLflow Registry**, predict, rollback.

Modos úteis:

| Comando | Quando |
|---------|--------|
| `./scripts/validate_platform.sh --skip-build` | Stack já levantada |
| `./scripts/validate_platform.sh --skip-build --skip-dag` | Pular espera DAG (~15 min) |
| `./scripts/validate_platform.sh --skip-build --infra-only` | Só contentores |
| `make validate-platform` | Build + validação completa |

- [ ] pytest anéis verde  
- [ ] validate_platform 0 FAIL  

---

## 9. Checklist rubrica TC02

Use esta tabela para fechar a entrega:

| # | Requisito TC02 | Como validar | OK |
|---|----------------|--------------|-----|
| R1 | Pipeline reprodutível (DVC) | `dvc repro` + `reports/recommendation/metrics.json` | [ ] |
| R2 | PyTorch (embedding) | `torch_embedding.pt` + champion no manifest | [ ] |
| R3 | Baselines comparados | `params.yaml` → popularity, nmf, torch_embedding | [ ] |
| R4 | MLflow tracking | UI experiment `tc02_recommendation` | [ ] |
| R5 | MLflow Model Registry | `tc02_recommender` alias Production | [ ] |
| R6 | API FastAPI autenticada | Swagger + JWT | [ ] |
| R7 | Orquestração Airflow | DAG `ml_training_dispatch` success | [ ] |
| R8 | Promote / deploy | `POST …/admin/promote` → 201 | [ ] |
| R9 | Inferência online | `POST …/predict` → `recommended_items` | [ ] |
| R10 | Rollback | `POST …/admin/rollback` após 2 deployments | [ ] |
| R11 | Métricas @K | NDCG, Hit Rate, etc. em metrics.json | [ ] |
| R12 | Docker compose integrado | `make docker-fresh` + stack healthy | [ ] |

---

## 10. Troubleshooting TC02

| Sintoma | Causa provável | Acção |
|---------|----------------|-------|
| `docker-fresh` lento | rebuild `--no-cache` | Normal na 1ª vez; aguardar |
| API JSONDecodeError no login | API ainda a boot | `sleep 10` e repetir |
| Train sync 502 | Worker down | `docker compose logs worker_recommendation` |
| Train sync event loop | API sem `WORKER_RECOMMENDATION_URL` | Rebuild API; compose deve ter URL do worker |
| Promote 400 ambiguidade | 2+ runs reco activos | Correr `validate_platform.sh` (dedupe) ou SQL em [`DOCUMENTACAO.md`](DOCUMENTACAO.md) §12 |
| Predict 404 | Sem deployment activo | Promote primeiro |
| Registry warning no promote | Cliente/servidor MLflow | Resposta 201 na BD OK; conferir UI passo 7 |
| DAG queued | DAG pausada | `airflow dags unpause ml_training_dispatch` |
| DVC falha preprocess | Falta `ratings.csv` | Secção 0.3 |
| MLflow vazio | Nenhum treino ainda | Passo 5.1 sync |
| pytest host falha asyncpg | Testes fora do venv | `make install-dev` ou usar só pytest anéis |

### SQL — desactivar runs reco duplicados

```bash
source .env
docker exec -e PGPASSWORD="$DATABASE_PASS" database_processing psql \
  -U "$DATABASE_USER" -d "$DATABASE_NAME" -c "
UPDATE pipeline_runs SET active = false
WHERE pipeline_type = 'recommendation'
  AND id != (SELECT MAX(id) FROM pipeline_runs
             WHERE pipeline_type = 'recommendation' AND status = 'completed');
"
```

---

## 11. Sessão copy-paste (resumo)

```bash
cd /home/gabriel/Machine-Learning
cp .env_example .env && echo "AIRFLOW_UID=$(id -u)" >> .env
# Editar .env: ENVIRONMENT=development

make install-dev
python3 scripts/validate_env.py
make docker-fresh
docker exec airflow_scheduler airflow dags unpause ml_training_dispatch

# DVC offline
make tc02-repro
cat reports/recommendation/metrics.json | python3 -m json.tool

# Swagger: http://localhost:8000/docs
# → auth → train/sync → promote → predict

# MLflow: http://localhost:5000 → tc02_recommender Production

# Gate automático
PYTHONPATH=src:. python3 -m pytest tests/platform_ring/ tests/ml_core_ring/ -q -o addopts=
VALIDATE_API_PASSWORD=admin1 ./scripts/validate_platform.sh --skip-build
```

---

## 12. Referências rápidas

| Recurso | Caminho / URL |
|---------|----------------|
| Swagger | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Params treino | `params.yaml` |
| Pipeline DVC | `dvc.yaml` |
| Registry model name | `tc02_recommender` |
| Experiment MLflow | `tc02_recommendation` |
| Script E2E | `scripts/validate_platform.sh` |
| Doc geral | `docs/DOCUMENTACAO.md` |

---

*Última actualização: roteiro TC02 — stack limpa, DVC, Swagger, Registry e validate_platform.*
