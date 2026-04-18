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
   - [4.4 Promover modelo (admin)](#44-promover-modelo-admin)
   - [4.5 Predição (consumidor)](#45-predição-consumidor)
5. [Contrato do CSV de treino](#5-contrato-do-csv-de-treino)
6. [Domínios disponíveis](#6-domínios-disponíveis)
7. [Monitoramento e manutenção](#7-monitoramento-e-manutenção)
8. [Estrutura do projeto](#8-estrutura-do-projeto)
9. [SLO definido](#9-slo-definido)
10. [Limitações conhecidas](#10-limitações-conhecidas)

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

### 4.4 Promover modelo (admin)

Torna um run concluído o modelo **ativo** para o domínio. Apenas modelos promovidos são usados em predições.

```bash
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "heart_disease",
    "pipeline_run_id": <ID_DO_RUN>
  }'
```

> `domain` deve ser idêntico ao `objective` usado no treino.

---

### 4.5 Predição (consumidor)

Qualquer usuário autenticado pode prever, desde que haja um modelo promovido para o domínio.

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
- `data/logs/<timestamp>/pipeline_<timestamp>.txt` — log de cada execução de pipeline

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

```
├── api/v1/endpoints/       # Rotas HTTP (auth, users, roles, processor)
├── core/                   # Config, auth, logging, middleware, deps
├── docker/                 # Dockerfiles (api, db, pgadmin)
├── docs/                   # Documentação e checklists
├── init_db/                # SQL de inicialização do banco
├── models/                 # ORM SQLAlchemy (users, roles, pipeline_runs, deployed_models, predictions)
├── schemas/                # Schemas Pydantic de entrada e saída
├── scripts/maintenance/    # Scripts offline: latency_report, drift_report
├── services/
│   ├── pipelines/          # Baseline, FeatureEngineering, strategies por domínio
│   └── processor/          # processor_service, deployment_service
├── docker-compose.yaml
├── main.py                 # Entrypoint FastAPI
└── .env_example            # Template de variáveis de ambiente
```

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
