# Documentacao da Solucao - Machine Learning Engineering

Documento tecnico-operacional do projeto. Ele consolida arquitetura, configuracao, contratos de API, pipelines de ML, persistencia, orquestracao, observabilidade e pontos de evolucao.

Complementos uteis:

- [`../README.md`](../README.md): guia rapido de setup e uso.
- [`MANUAL_DO_USUARIO.md`](MANUAL_DO_USUARIO.md): fluxo em linguagem funcional.
- [`OBSERVABILIDADE_E_MANUTENCAO.md`](OBSERVABILIDADE_E_MANUTENCAO.md): logs, latencia e drift.
- [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md): EDA, metricas e discussoes tecnicas.
- [`CHECKLIST_PROJETO.md`](CHECKLIST_PROJETO.md): pendencias e evolucoes.

---

## 1. Visao geral

O projeto e uma plataforma de Machine Learning Engineering para **classificacao binaria tabular**. O fluxo principal e:

```text
CSV de treino
  -> Pipeline Baseline
  -> Pipeline Feature Engineering
  -> registro em pipeline_runs
  -> promocao em deployed_models
  -> predicao via API
  -> auditoria em predictions
  -> monitoramento offline de latencia e drift
```

### 1.1 Objetivos

- Treinar modelos baseline e modelos com engenharia de features.
- Registrar runs, metricas, artefatos e erros.
- Promover um modelo ativo por dominio.
- Servir predicoes autenticadas.
- Manter historico de deployments e permitir rollback.
- Coletar logs de requisicao e gerar relatorios offline.

### 1.2 Stack

| Camada | Tecnologia |
|--------|------------|
| API | FastAPI, Pydantic v2 |
| Auth | JWT Bearer, `python-jose`, `passlib` |
| Persistencia | PostgreSQL, SQLAlchemy async, `asyncpg` |
| ML | pandas, scikit-learn, MLflow, joblib |
| Orquestracao | Apache Airflow |
| Infra local | Docker Compose |
| Observabilidade | logs JSONL, Dozzle, scripts de manutencao |

---

## 2. Estado atual dos dominios

O codigo tem estrutura para multiplos dominios, mas ha desalinhamentos importantes:

| Componente | Estado atual |
|------------|--------------|
| `MLDomain` | `heart_disease`, `churn` |
| `STRATEGY_REGISTRY` | registra apenas `heart_disease` |
| Rotas HTTP de treino | `objective: Literal["churn"]` |
| Schema de predicao | features estritas de `heart_disease` |
| Labels de classe | `heart_disease` e `churn` em `CLASS_LABELS` |

Impacto:

- O fluxo de Feature Engineering exige strategy registrada. Hoje isso existe para `heart_disease`.
- As rotas HTTP de treino aceitam, pela validacao da API, apenas `churn`; nesse estado, o FE tende a falhar por falta de `ChurnFeatures`.
- A predicao esta tipada para `heart_disease`, mesmo com enum contendo `churn`.

Antes de demonstrar o fluxo completo por HTTP, alinhar uma destas alternativas:

1. Permitir `heart_disease` nas rotas de treino; ou
2. Implementar e registrar `ChurnFeatures`; e
3. Ajustar `PredictRequest` para suportar schemas por dominio.

---

## 3. Arquitetura

### 3.1 Camadas

| Camada | Local | Responsabilidade |
|--------|-------|------------------|
| Entrada HTTP | `api/v1/endpoints/` | Rotas, status codes, dependencias de auth/admin/ambiente |
| Contratos | `schemas/` | Modelos Pydantic, enums e responses |
| Servicos | `services/processor/` | Treino, promocao, rollback, predicao e bundles |
| Pipelines ML | `services/pipelines/` | Baseline, FE, selecao/tuning, strategies |
| Persistencia | `models/` | ORM das tabelas |
| Configuracao | `core/configs.py` | Variaveis de ambiente e paths |
| Auth/deps | `core/auth.py`, `core/deps.py` | JWT, usuario atual, gates admin/producao |
| Logs | `core/logging_*`, `core/custom_logger.py` | Logging raiz, HTTP e pipelines |
| Airflow | `airflow/dags/ml_training_pipeline.py` | Orquestracao manual de treino |
| Manutencao | `scripts/maintenance/` | Latencia e drift offline |

### 3.2 Fluxo de predicao

```text
POST /processor/predict
  -> valida JWT
  -> valida body Pydantic
  -> busca deployed_models active por domain
  -> carrega joblib de pipeline_runs.model_path
  -> prepara/alinha features
  -> model.predict / predict_proba
  -> grava predictions
  -> retorna PredictResponse
```

### 3.3 Fluxo de treino sincrono

```text
POST /processor/admin/train/baseline
  -> salva CSV em PATH_DATA
  -> cria pipeline_runs(status=processing)
  -> executa Baseline
  -> salva modelo/CSV/metricas
  -> atualiza status completed ou failed

POST /processor/admin/train/feature-engineering
  -> salva CSV temporario
  -> cria pipeline_runs(status=processing)
  -> executa Baseline interno
  -> executa FeatureEngineering
  -> monta ZIP de artefatos
  -> grava model_path e metricas no run
```

---

## 4. Configuracao

As configuracoes sao carregadas por `Settings` em `core/configs.py`, a partir de `.env`.

### 4.1 Variaveis essenciais

| Variavel | Descricao |
|----------|-----------|
| `PROJECT_NAME` | Titulo da aplicacao FastAPI |
| `PROJECT_VERSION` | Prefixo das rotas, por exemplo `/v1` |
| `DATABASE_USER`, `DATABASE_PASS`, `DATABASE_SERVER`, `DATABASE_PORT`, `DATABASE_NAME` | Conexao PostgreSQL |
| `SECRET`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT |
| `TEST_SIZE`, `RANDOM_STATE` | Parametros globais dos splits |
| `PATH_DATA`, `PATH_DATA_PREPROCESSED`, `PATH_MODEL`, `PATH_GRAPHS`, `PATH_LOGS` | Paths de artefatos |
| `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_ROOT` | Tracking local do MLflow |
| `ENVIRONMENT` | Controla gates de producao |
| `SYNC_FE_TUNE_MAX_MINUTES` | Teto de tuning em rotas sincronas |
| `PATH_API_REQUEST_LOGS` | Destino dos logs HTTP JSONL |
| `PATH_MAINTENANCE_REPORTS` | Saida dos relatorios |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USER`, `AIRFLOW_PASSWORD` | Integracao com Airflow REST |
| `ML_SHARED_PATH` | Volume compartilhado API/Airflow para uploads |

### 4.2 Ambientes

`settings.is_production` retorna verdadeiro quando `ENVIRONMENT` normalizado for:

```text
prd, prod, production
```

Nesses ambientes:

- `/processor/admin/train/baseline` retorna `403`;
- `/processor/admin/train/feature-engineering` retorna `403`;
- `/processor/admin/train/trigger-dag` retorna `403`;
- `/processor/predict`, promocao, historico e rollback continuam disponiveis conforme permissao.

---

## 5. Banco de dados

O schema inicial esta em `init_db/database.sql`.

### 5.1 Tabelas principais

| Tabela | Funcao |
|--------|-------|
| `roles` | Perfis do sistema (`User`, `Administrator`) |
| `users` | Usuarios autenticados |
| `pipeline_runs` | Execucoes de treino, metricas, paths e status |
| `deployed_models` | Modelo ativo/arquivado por dominio |
| `predictions` | Auditoria de predicoes realizadas |

### 5.2 Campos importantes

`pipeline_runs`:

- `pipeline_type`: `baseline` ou `feature_engineering`.
- `objective`: dominio usado no treino.
- `status`: `processing`, `completed` ou `failed`.
- `model_path`: arquivo joblib usado para promocao/predicao.
- `csv_output_path`: CSV gerado pelo baseline, quando aplicavel.
- `metrics`: JSON de metricas.
- `error_message`: erro truncado em falhas.
- `active`: usado para elegibilidade de promocao.

`deployed_models`:

- `domain`: dominio canonicamente normalizado.
- `pipeline_run_id`: run promovido.
- `status`: `active` ou `archived`.
- `metrics_snapshot`: copia das metricas no momento da promocao.

Ha um indice unico parcial para impedir mais de um deployment ativo por dominio:

```sql
CREATE UNIQUE INDEX uq_deployed_models_one_active_per_domain
ON public.deployed_models (domain)
WHERE status = 'active' AND active IS TRUE;
```

---

## 6. Autenticacao e autorizacao

### 6.1 Rotas de auth

Prefixo: `/v1/auth`.

| Metodo | Rota | Descricao |
|--------|------|-----------|
| `POST` | `/signup` | Cria usuario comum |
| `POST` | `/authenticate` | Retorna JWT Bearer |
| `GET` | `/logged` | Retorna usuario atual |

`signup` ignora promocao de papel no payload: novos usuarios ficam com `role_id=1` por padrao do modelo.

### 6.2 Papeis

| Papel | ID | Uso |
|-------|----|-----|
| Usuario | `1` | Pode autenticar e predizer |
| Administrador | `2` | Pode treinar, promover, rollback e gerenciar roles |

Rotas administrativas usam `require_admin`.

---

## 7. API do processador

Prefixo: `/v1/processor`.

| Metodo | Rota | Permissao | Retorno |
|--------|------|-----------|---------|
| `POST` | `/predict` | Usuario autenticado | `PredictResponse` |
| `POST` | `/admin/promote` | Admin | `DeployedModelResponse` |
| `POST` | `/admin/rollback` | Admin | `DeployedModelResponse` |
| `GET` | `/admin/deployments/{domain}/history` | Admin | Lista de deployments |
| `POST` | `/admin/train/baseline` | Admin + nao prod | CSV (`FileResponse`) |
| `POST` | `/admin/train/feature-engineering` | Admin + nao prod | ZIP (`FileResponse`) |
| `POST` | `/admin/train/trigger-dag` | Admin + nao prod | `TriggerDagResponse` |

### 7.1 Predicao

Request:

```json
{
  "domain": "heart_disease",
  "features": {
    "age": 55,
    "trestbps": 140,
    "chol": 250,
    "fbs": false,
    "thalch": 150,
    "exang": false,
    "oldpeak": 1.5,
    "ca": 0,
    "sex_Male": true,
    "cp_atypical angina": false,
    "cp_non-anginal": true,
    "cp_typical angina": false,
    "restecg_normal": true,
    "restecg_st-t abnormality": false,
    "slope_flat": false,
    "slope_upsloping": true,
    "thal_normal": true,
    "thal_reversable defect": false
  }
}
```

Notas:

- `features` usa `extra="forbid"`; chaves extras geram `422`.
- Campos com espaco ou hifen usam aliases Pydantic.
- A probabilidade e convertida para percentual (`0.82` vira `82.0`).
- Se nao houver deployment ativo, a API retorna `404`.

### 7.2 Promocao

Request:

```json
{
  "domain": "heart_disease",
  "pipeline_run_id": 123
}
```

Validacoes em `deployment_service.promote_pipeline_run`:

- run existe e esta ativo;
- `status == "completed"`;
- `run.objective` coincide com `domain`;
- `model_path` existe no filesystem;
- deployment ativo anterior e arquivado.

### 7.3 Rollback

Request:

```json
{
  "domain": "heart_disease"
}
```

Reativa o deployment `archived` mais recente do dominio e arquiva o atual.

### 7.4 Treino baseline

Multipart:

| Campo | Tipo | Descricao |
|-------|------|-----------|
| `file` | CSV | Dataset de treino |
| `objective` | string | No codigo atual, validado como `churn` |

Resposta:

- corpo: CSV pre-processado;
- headers:
  - `X-Pipeline-Run-Id`;
  - `X-Pipeline-Type`;
  - `X-Pipeline-Objective`;
  - `X-Pipeline-Metrics`.

### 7.5 Treino Feature Engineering

Multipart:

| Campo | Tipo | Padrao |
|-------|------|--------|
| `file` | CSV | obrigatorio |
| `objective` | string | validado atualmente como `churn` |
| `optimization_metric` | enum | `accuracy` |
| `min_precision` | float opcional | `null` |
| `min_roc_auc` | float opcional | `null` |
| `time_limit_minutes` | int | `2` |
| `acc_target` | float opcional | `null` |

Metricas aceitas:

```text
accuracy, precision, recall, f1, roc_auc
```

Resposta:

- corpo: ZIP com artefatos;
- headers `X-Pipeline-*`.

---

## 8. Pipelines de ML

### 8.1 Baseline

Classe: `services/pipelines/baseline.py::Baseline`.

Responsabilidades:

- carregar CSV de caminho explicito;
- detectar target como ultima coluna;
- recusar target nulo;
- converter colunas `object` que parecam numericas;
- binarizar target (`> 0` vira `1`);
- separar treino/teste com `train_test_split`;
- imputar numericos/categoricos dentro do pipeline sklearn;
- one-hot encoding para categoricas;
- treinar `LogisticRegression(class_weight="balanced")`;
- salvar modelo e CSV de amostra;
- registrar metricas e artefatos no MLflow quando possivel.

Contrato:

- classificacao binaria;
- sem coluna de ID;
- target na ultima coluna;
- cabecalho obrigatorio.

### 8.2 Feature Engineering

Classe: `services/pipelines/feature_engineering.py::FeatureEngineering`.

Responsabilidades:

- carregar CSV pre-processado via manifest do baseline;
- validar dominio por `FeatureStrategy`;
- criar features especificas do dominio;
- selecionar features com `SelectKBest(f_classif)`;
- comparar modelos:
  - Decision Tree;
  - Random Forest;
  - SVM RBF;
  - Gradient Boosting;
- aplicar guardrails opcionais (`min_precision`, `min_roc_auc`);
- tuning com `ParameterSampler` e budget de tempo;
- calcular importancia de features;
- salvar melhor pipeline em joblib;
- registrar metricas e figuras no MLflow.

### 8.3 Strategy de dominio

Contrato base: `services/pipelines/feature_strategies/base.py::FeatureStrategy`.

Metodos obrigatorios:

- `required_columns()`;
- `created_features()`;
- `build(df)`.

Strategy implementada:

| Dominio | Classe | Colunas minimas |
|---------|--------|-----------------|
| `heart_disease` | `HeartDiseaseFeatures` | `age`, `chol` |

Features criadas para `heart_disease` incluem:

- `age_squared`;
- `cholesterol_to_age`;
- `max_hr_pct`;
- `bp_chol_ratio`;
- `fbs_flag`;
- `exang_flag`;
- `stress_index`;
- `age_decade`;
- `risk_interaction`;
- `high_st_depression_flag`.

---

## 9. Airflow

DAG: `airflow/dags/ml_training_pipeline.py`.

### 9.1 Tasks

```text
validate_input -> run_baseline -> run_fe -> notify_complete
```

| Task | Funcao |
|------|-------|
| `validate_input` | Valida existencia do CSV e colunas minimas da strategy |
| `run_baseline` | Executa `Baseline` |
| `run_fe` | Executa `FeatureEngineering` |
| `notify_complete` | Loga resumo e proximo passo de promocao |

### 9.2 Parametros

O DAG combina Airflow Variables com `dag_run.conf`. O `dag_run.conf` tem prioridade.

Variable principal:

```text
ml_training_pipeline_conf
```

Exemplo:

```json
{
  "objective": "heart_disease",
  "csv_path": "/opt/airflow/ml_shared/uploads/dados.csv",
  "optimization_metric": "accuracy",
  "time_limit_minutes": 30,
  "acc_target": 0.85,
  "user_id": 1
}
```

### 9.3 Observacao operacional

O DAG importa modulos do projeto a partir de `ML_PROJECT_ROOT`. Em ambiente containerizado, garanta que o codigo do projeto esteja visivel ao worker Airflow nesse caminho, alem do CSV compartilhado. Caso contrario, imports como `services.pipelines...` falham.

---

## 10. Observabilidade

### 10.1 Logs HTTP

Middleware: `core/middleware/request_record.py`.

Destino padrao:

```text
logs/api_requests/access.jsonl
```

Cada linha contem dados como:

- timestamp;
- metodo;
- path;
- status;
- duracao em ms;
- request_id;
- cliente;
- erro, se houver.

### 10.2 Logs de pipeline

Destino padrao:

```text
data/logs/<timestamp>/pipeline_<timestamp>.txt
```

O logger principal e `ml.pipeline`.

### 10.3 Relatorio de latencia

Script:

```bash
python scripts/maintenance/latency_report.py --slo-ms 300
```

Saida: CSV em `artifacts/reports/`.

SLO atual:

| Rota | Metrica | Limite |
|------|---------|--------|
| `POST /processor/predict` | p95 | `< 300ms` |

### 10.4 Relatorio de drift

Script:

```bash
python scripts/maintenance/drift_report.py \
  --train-csv data/referencia.csv \
  --predictions-csv exports/predictions.csv
```

Usa PSI:

| PSI | Status |
|-----|--------|
| `< 0.10` | `ok` |
| `0.10 - 0.25` | `warning` |
| `> 0.25` | `critical` |

O drift nao roda em tempo real no endpoint de predicao.

---

## 11. Criterio de promocao recomendado

O codigo valida integridade minima do run e do artefato. A regra de negocio recomendada para decidir promocao e:

| Criterio | Recomendacao |
|----------|--------------|
| Run concluido | Obrigatorio |
| Artefato em `model_path` | Obrigatorio |
| Dominio igual ao objective | Obrigatorio |
| Metrica principal | Candidato deve superar ativo, idealmente >= 2% |
| Drift PSI medio | Idealmente `< 0.10` |
| Latencia | p95 de `/predict` dentro do SLO |

Processo:

1. Treinar modelo candidato.
2. Conferir metricas do run.
3. Comparar com historico de deployments.
4. Rodar drift/latencia quando houver dados suficientes.
5. Promover.
6. Monitorar.
7. Fazer rollback em caso de regressao.

---

## 12. Como adicionar um novo dominio

Checklist tecnico:

1. Adicionar o valor em `MLDomain` (`schemas/processor_schemas.py`).
2. Criar uma classe em `services/pipelines/feature_strategies/`.
3. Implementar `required_columns`, `created_features` e `build`.
4. Registrar a classe em `STRATEGY_REGISTRY`.
5. Adicionar labels em `CLASS_LABELS`.
6. Ajustar as rotas de treino para aceitar o novo `objective`.
7. Criar schema Pydantic de predicao para o dominio.
8. Ajustar `PredictRequest` para `Union` discriminada ou endpoint por dominio.
9. Criar exemplos de payload e CSV.
10. Testar: baseline -> FE -> promote -> predict -> rollback.

---

## 13. Debitos e riscos conhecidos

| Item | Impacto | Acao sugerida |
|------|---------|---------------|
| Rotas de treino aceitam apenas `churn`, mas FE registra `heart_disease` | Quebra fluxo HTTP de FE | Alinhar `Literal` com `MLDomain` e registry |
| `churn` sem `ChurnFeatures` | FE de churn falha | Implementar strategy |
| Predicao tipada apenas para heart disease | Multi-dominio incompleto | Usar `Union` por dominio ou endpoints separados |
| Airflow pode nao enxergar codigo do projeto no container | DAG falha em imports | Montar/copy do projeto para `ML_PROJECT_ROOT` |
| Drift e latencia agregada sao offline | Sem alertas automaticos | Agendar scripts ou integrar APM |
| MLflow local SQLite | Limitado para equipe/producao | Usar tracking server compartilhado |
| Criterios de promocao nao automatizados | Admin pode promover sem gate de negocio | Implementar validacoes de metrica/drift no promote |

---

## 14. Referencia de arquivos

| Arquivo | Papel |
|---------|------|
| `main.py` | Inicializa FastAPI e inclui routers |
| `core/configs.py` | Settings e `database_url` |
| `api/v1/api.py` | Agrega routers |
| `api/v1/endpoints/processor.py` | Rotas de treino, predict, promote, rollback |
| `services/processor/processor_service.py` | Orquestracao de treino/predicao |
| `services/processor/deployment_service.py` | Regras de promocao e rollback |
| `services/pipelines/baseline.py` | Pipeline baseline |
| `services/pipelines/feature_engineering.py` | Pipeline FE |
| `services/pipelines/feature_strategies/` | Strategies por dominio |
| `schemas/processor_schemas.py` | Contratos do processador |
| `init_db/database.sql` | Schema inicial |
| `docker-compose.yaml` | Ambiente local completo |
| `airflow/dags/ml_training_pipeline.py` | DAG de treino |
| `scripts/maintenance/` | Relatorios operacionais |
