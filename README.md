# ML Engineering — Pipeline de Classificação Binária

Plataforma de Machine Learning Engineering para treinamento, versionamento e inferência de modelos de **classificação binária tabulada** (ex.: Heart Disease, Churn).

Cobre o ciclo completo: ingestão de dados → baseline → feature engineering → promoção de modelo → API de predição → monitoramento.

**Documentação consolidada (arquitetura, ambientes dev/prod, contratos de API, débitos, novos domínios):** [`docs/DOCUMENTACAO_SOLUCAO.md`](docs/DOCUMENTACAO_SOLUCAO.md).

---

## Sumário

1. [Pré-requisitos](#1-pré-requisitos)
2. [Setup do ambiente](#2-setup-do-ambiente)
3. [Subir os serviços](#3-subir-os-serviços)
4. [Fluxo completo — passo a passo](#4-fluxo-completo--passo-a-passo)
   - [4.1 Criar conta e autenticar](#41-criar-conta-e-autenticar)
   - [4.2 Treino Baseline (admin)](#42-treino-baseline-admin)
   - [4.3 Treino Feature Engineering (admin)](#43-treino-feature-engineering-admin)
   - [4.4 Listar pipeline runs (admin)](#44-listar-pipeline-runs-admin)
   - [4.5 Promover modelo (admin)](#45-promover-modelo-admin)
   - [4.6 Predição (consumidor)](#46-predição-consumidor)
5. [Contrato do CSV de treino](#5-contrato-do-csv-de-treino)
6. [Domínios disponíveis](#6-domínios-disponíveis)
7. [Monitoramento e manutenção](#7-monitoramento-e-manutenção)
8. [Estrutura do projeto](#8-estrutura-do-projeto)
   - [8.1 Pastas do Tech Challenge e deste repo](#81-pastas-do-tech-challenge-e-deste-repo)
9. [SLO definido](#9-slo-definido)
10. [Critério de promoção de modelos](#10-critério-de-promoção-de-modelos)
11. [Limitações conhecidas](#11-limitações-conhecidas)

---

## 1. Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.20
- Porta `8000` (API), `5432` (Postgres), `5050` (pgAdmin), `8888` (Dozzle) livres

---

## 2. Setup do ambiente

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd Machine-Learning

# 2. Criar o arquivo de variáveis de ambiente
cp .env_example .env
```

Abrir o `.env` e preencher **obrigatoriamente**:

| Variável | Descrição |
|----------|-----------|
| `DATABASE_USER` | Usuário do PostgreSQL |
| `DATABASE_PASS` | Senha do PostgreSQL |
| `DATABASE_NAME` | Nome do banco (ex.: `processing`) |
| `SECRET` | Chave JWT — gerar com `python -c "import secrets; print(secrets.token_hex(32))"` |
| `PGADMIN_EMAIL` | E-mail de acesso ao pgAdmin |
| `PGADMIN_PASSWORD` | Senha do pgAdmin |
| `PROJECT_NAME` | Nome do projeto (ex.: `ML-Engineering`) |
| `PROJECT_VERSION` | Prefixo da API (ex.: `/v1`) |

> `DATABASE_SERVER` já está configurado como `db_processing` no compose — não alterar para uso com Docker.

---

## 3. Subir os serviços

```bash
docker compose up --build
```

| Serviço | URL | Descrição |
|---------|-----|-----------|
| API | http://localhost:8000/docs | Swagger — documentação interativa |
| Dozzle | http://localhost:8888 | Logs de todos os containers em tempo real |
| pgAdmin | http://localhost:5050 | Interface web do banco de dados |

Para parar:
```bash
docker compose down
```

Para parar e remover volumes (reset completo):
```bash
docker compose down -v
```

---

## 4. Fluxo completo — passo a passo

Todos os exemplos usam `curl`. O mesmo fluxo pode ser executado pelo Swagger em `/docs`.

### 4.1 Criar conta e autenticar

**Criar conta:**
```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name": "Seu Nome", "email": "usuario@exemplo.com", "password": "sua_senha"}'
```

**Login — obter token:**
```bash
curl -X POST http://localhost:8000/v1/auth/authenticate \
  -F "username=usuario@exemplo.com" \
  -F "password=sua_senha"
```

Resposta:
```json
{ "access_token": "<TOKEN>", "token_type": "bearer" }
```

> Guardar o `access_token` — será usado em todas as rotas como `Authorization: Bearer <TOKEN>`.

---

### 4.2 Treino Baseline (admin)

> Requer conta com papel **Administrador**.

**Contrato do CSV:** ver [seção 5](#5-contrato-do-csv-de-treino).

```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/baseline \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@caminho/para/dados.csv" \
  -F "objective=heart_disease"
```

A resposta inclui o CSV pré-processado no corpo e os metadados nos cabeçalhos `X-Pipeline-*`.

Anotar o `pipeline_run_id` retornado nos cabeçalhos — necessário para promoção.

---

### 4.3 Treino Feature Engineering (admin)

Usar o CSV pré-processado devolvido pelo baseline como entrada.

```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/feature-engineering \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@caminho/para/dados_preprocessados.csv" \
  -F "objective=heart_disease" \
  -F "optimization_metric=recall"
```

Métricas disponíveis: `accuracy` · `precision` · `recall` · `f1` · `roc_auc`

Anotar o `pipeline_run_id` do cabeçalho de resposta.

---

### 4.4 Listar pipeline runs (admin)

Lista execuções (`pipeline_runs`) para escolher o `pipeline_run_id` na promoção, com filtros opcionais (`domain` = **apenas `churn`** se quiser filtrar, `pipeline_type`, `status`, `limit`). Resposta ordenada por `created_at` (mais recentes primeiro).

```bash
curl -s -X GET "http://localhost:8000/v1/processor/admin/runs?domain=churn&pipeline_type=feature_engineering&status=completed&limit=20" \
  -H "Authorization: Bearer <TOKEN>"
```

Para ver todos os runs recentes sem filtro de domínio, omita os query params ou use apenas `limit`.

---

### 4.5 Promover modelo (admin)

Torna um run concluído o modelo **ativo** para o domínio. Apenas modelos promovidos são usados em predições.

**Quem pode ser promovido:** apenas runs de **`feature_engineering`** com `status` concluído (`pipeline_type` no formulário). Runs só de baseline **não** entram em `/predict` por este fluxo.

```bash
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer <TOKEN>" \
  -F "domain=churn" \
  -F "pipeline_run_id=<ID_DO_RUN>" \
  -F "pipeline_type=feature_engineering"
```

> `domain` deve ser idêntico ao `objective` usado no treino (hoje o promote está alinhado ao fluxo **churn** no Swagger).

#### O que a rota `/predict` carrega de facto (sklearn vs PyTorch)

| Aspeto | Comportamento neste projeto |
|--------|-----------------------------|
| **Artefacto em produção** | Ficheiro **`.joblib`** gerado pelo FE: `sklearn.pipeline.Pipeline` (pré-processamento + modelo selecionado no tuning, ex. Random Forest). A API faz `joblib.load` e usa `predict` / `predict_proba`. |
| **MLP (PyTorch)** | Treinada no mesmo pipeline de FE para **comparação**; métricas e artefactos opcionais vão para o **MLflow**. **Não** é este tensor o que o `POST /predict` carrega hoje. Servir a MLP exigiria outro desenho (ex. TorchScript/ONNX + alinhamento ao mesmo pré-processamento). |
| **Rastreio** | O run promovido continua ligado a um `pipeline_run_id`; o joblib apontado em `model_path` é o que vale para inferência. |

---

### 4.6 Predição (consumidor)

Qualquer usuário autenticado pode prever, desde que haja um modelo promovido para o domínio.

O corpo segue o **domínio** e o schema de `features` definidos no Swagger (ex. **churn**: atributos brutos alinhados ao treino; não enviar colunas pós–one-hot). O modelo executado é sempre o **Pipeline sklearn** do run promovido, conforme a tabela acima.

```bash
curl -X POST http://localhost:8000/v1/processor/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "heart_disease",
    "features": {
      "age": 55,
      "sex": 1,
      "cp": 2,
      "trestbps": 140,
      "chol": 250,
      "fbs": 0,
      "restecg": 1,
      "thalch": 150,
      "exang": 0,
      "oldpeak": 1.5,
      "slope": 2,
      "ca": 0,
      "thal": 2
    }
  }'
```

Resposta:
```json
{
  "id": 1,
  "domain": "heart_disease",
  "pipeline_run_id": 3,
  "prediction": 1,
  "probability": 0.82,
  "input_data": { ... }
}
```

---

## 5. Contrato do CSV de treino

| Regra | Detalhe |
|-------|---------|
| **Última coluna = target** | O pipeline detecta automaticamente a coluna alvo pela posição |
| **Sem coluna de ID** | Remover antes do upload (ex.: `customer_id`, `patient_id`) |
| **Tipo de problema** | Apenas **classificação binária** — o target é binarizado (qualquer valor > 0 vira 1) |
| **Valores ausentes** | Imputados automaticamente: mediana para numéricos, moda para categóricos |
| **Formato** | `.csv` com cabeçalho na primeira linha |

---

## 6. Domínios disponíveis

| `objective` / `domain` | Dataset de referência | Features obrigatórias |
|------------------------|-----------------------|-----------------------|
| `heart_disease` | UCI Heart Disease | `age`, `chol` (mínimo) |
| `churn` | *(em implementação)* | *(a definir)* |

> Para adicionar um novo domínio: implementar `FeatureStrategy` em `services/pipelines/feature_strategies/` e registrar em `__init__.py`.

---

## 7. Monitoramento e manutenção

### Logs em tempo real
Acesse **http://localhost:8888** (Dozzle) para visualizar os logs de todos os containers.

Logs persistidos em volume:
- `logs/api_requests/access.jsonl` — uma linha JSON por requisição HTTP
- `src/data/logs/<timestamp>/pipeline_<timestamp>.txt` — log de cada execução de pipeline (treinos sob `PATH_DATA`/`PATH_LOGS`)

### Relatório de latência
```bash
# Dentro do container da API ou com o ambiente local configurado:
python scripts/maintenance/latency_report.py --slo-ms 300
```
Gera CSV em `artifacts/reports/latency_summary_<timestamp>.csv` com p50/p90/p95/p99 por rota e status do SLO.

### Relatório de drift
```bash
# 1. Exportar predições do banco:
# COPY (SELECT id, pipeline_run_id, input_data, prediction, probability FROM predictions)
# TO '/tmp/predictions.csv' WITH CSV HEADER;

# 2. Rodar o relatório:
python scripts/maintenance/drift_report.py \
  --train-csv data/heart_disease_reference.csv \
  --predictions-csv exports/predictions.csv
```
Gera CSV em `artifacts/reports/drift_psi_<timestamp>.csv` com PSI por feature e status `ok / warning / critical`.

### Processo de decisão

| Situação | Ação |
|----------|------|
| Latência p95 > SLO (`breach`) | Investigar modelo pesado ou infra; considerar retreino mais leve |
| Drift PSI `warning` (0.10–0.25) | Aumentar frequência de monitoramento |
| Drift PSI `critical` (> 0.25) | Coletar dados recentes → retreinar → promover novo run → reavaliar drift |

**Periodicidade recomendada:** após cada deploy e semanalmente com volume de predições ativo.

---

## 8. Estrutura do projeto

O **código FastAPI + pipelines ML** está sob **`src/`** (equivalente ao `src/` pedido em materiais tipo Tech Challenge). Docker, Airflow e `docker-compose` ficam na **raiz** do repositório.

```
├── src/
│   ├── api/v1/endpoints/   # Rotas HTTP (auth, users, roles, processor, health)
│   ├── core/               # Config, auth, logging, middleware, deps
│   ├── models/             # ORM SQLAlchemy
│   ├── schemas/            # Pydantic
│   ├── services/           # Pipelines ML, processor, auth
│   ├── data/               # CSVs, pre_processed, logs de pipeline (opcional no VCS)
│   ├── artifacts/          # MLflow (mlruns), modelos .joblib, relatórios
│   ├── graphs/             # Gráficos exportados pelo baseline/FE
│   └── scripts/maintenance/
├── airflow/                # DAGs (fora de src)
├── docker/
├── docs/
├── init_db/
├── main.py                 # FastAPI — faz `sys.path` → `src/`
├── docker-compose.yaml     # API: volumes em /var/www/src/... e /var/www/logs
└── .env_example
```

### 8.1 Pastas do Tech Challenge e deste repo

| Pedido habitual (TC) | Onde está aqui |
|----------------------|----------------|
| `src/` (código) | **`src/`** — `api/`, `core/`, `services/`, `schemas/` (+ ORM em `src/models/`) |
| `data/` | **`src/data/`** por omissão (`PATH_*` no `.env`) |
| `models/` (pesos ML) | **`src/artifacts/models/`** (`PATH_MODEL`) — não confundir com `src/models/` (ORM) |
| `tests/` | Outro membro (`pytest` planeado) |
| `notebooks/` | Opcional (`reference/`, etc.) |
| `docs/` | `docs/` na raiz |

---

## 9. SLO definido

| Rota | Métrica | Threshold |
|------|---------|-----------|
| `POST /processor/predict` | p95 de latência | < 300ms |

Verificar com: `python scripts/maintenance/latency_report.py --slo-ms 300`

---

## 10. Critério de promoção de modelos

Um modelo só deve ser promovido para produção quando **todas** as condições abaixo forem atendidas:

| Critério | Threshold |
|----------|-----------|
| Métrica principal do novo run > modelo ativo | ≥ +2% |
| PSI médio das features (drift) | < 0.10 |
| Run com `status = completed` e artefato existente | Obrigatório |

**Processo:**
1. Rodar `drift_report.py` com CSV de treino e export de predições recentes.
2. Confirmar PSI médio abaixo de 0.10.
3. Comparar métricas do run candidato com `GET /processor/admin/deployments/{domain}/history`.
4. Se aprovado: `POST /processor/admin/promote`.
5. Em caso de problema após promoção: `POST /processor/admin/rollback`.

---

## 11. Limitações conhecidas

- **Escopo:** apenas classificação binária tabulada. Regressão e multiclasse não suportados.
- **Desbalanceamento:** tratado via `class_weight='balanced'` na Regressão Logística do Baseline. Estratégias de reamostragem (SMOTE) não implementadas.
- **Drift:** monitoramento offline (script manual/agendado), sem alertas em tempo real no predict.
- **Orquestração:** treinos executados na thread HTTP. Integração com Airflow prevista — ver `docs/CHECKLIST_PROJETO.md` Bloco 3.
- **MLflow:** tracking local (SQLite). Para uso em equipe, configurar servidor MLflow externo via `MLFLOW_TRACKING_URI`.

+------------------------------------------------------------------------------------------+
|                              MÁQUINA / DOCKER HOST                                        |
|                                                                                          |
|  +------------------+       +----------------------------------------------------------+|
|  | Cliente          |       |  Rede Docker (ex.: nwprocessing)                        ||
|  | (Browser / curl) |       |                                                           ||
|  +--------+---------+       |  +-------------------------+     +----------------------+  ||
|           |                 |  | API FastAPI (:8000)      |     | PostgreSQL (:5432)    |  ||
|           | HTTPS/HTTP      |  | main.py + pacote em src/ |     | users, roles, runs,   |  ||
|           v                 |  |  /v1/auth  (JWT)         |     | deployments, predicts |  ||
|  +------------------+      |  |  /v1/processor          |<--->|                       |  ||
|  | Swagger / Insomnia   |   |  |    train, runs,        |     +----------------------+  ||
|  +------------------+      |  |    promote, predict      |                               ||
|                             |  |  /v1/health            |                               ||
|                             |  +------------+------------+                               ||
|                             |               |                                             ||
|                             |               v                                             ||
|                             |  +-------------------------+    +-------------------------+  ||
|                             |  | Volumes API             |    | Airflow                 |  ||
|                             |  |  src/data               |    |  Web :8080 / Scheduler  |  ||
|                             |  |  src/artifacts (joblib,|    |  DAG: Baseline -> FE    |  ||
|                             |  |            mlruns…)     |    |  Imports: pasta src     |  ||
|                             |  |  logs/api_requests      |    |    montada (ml_code)    |  ||
|                             |  |  ml_shared (upload CSV) |<-->+  Volume ml_project       |  ||
|                             |  |                         |    |  (mesmo vol. que API    |  ||
|                             |  |                         |    |   chama ml_shared)      |  ||
|                             |  +-------------------------+    +-------------------------+  ||
|                             |                                                             ||
|                             |  +----------+    +----------+  (opcional)                  ||
|                             |  | pgAdmin  |    | Dozzle   |                               ||
|                             |  | :5050    |    | :8888    |                               ||
|                             |  +----------+    +----------+                               ||
|                             +----------------------------------------------------------+|
+------------------------------------------------------------------------------------------+

Fluxo de negócio (resumo):

  CSV  --->  [ Baseline ]  --->  manifest + sample  --->  [ Feature Engineering ]
                |                                              |
                +------------ MLflow (métricas, artefactos) --+
                |
  Admin  --->  POST promote (run FE)  --->  modelo ativo no domínio
                |
  Utilizador -> POST predict  --->  joblib + resposta (predição / prob)

Legenda:
  <-->  lê/escreve na mesma base ou no mesmo volume conforme configuração
  API   trabalho síncrono; Airflow orquestra o mesmo tipo de pipelines no worker