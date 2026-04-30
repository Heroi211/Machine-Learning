# ML Engineering - Pipeline de Classificacao Binaria

Plataforma de Machine Learning Engineering para treinamento, versionamento, promocao e inferencia de modelos de **classificacao binaria em dados tabulares**.

O projeto cobre o fluxo:

```text
CSV de treino -> Baseline -> Feature Engineering -> PipelineRuns
              -> Promocao de modelo -> API de predicao -> Monitoramento offline
```

Documentacao complementar:

- [Documentacao consolidada da solucao](docs/DOCUMENTACAO_SOLUCAO.md)
- [Manual do usuario](docs/MANUAL_DO_USUARIO.md)
- [Observabilidade e manutencao](docs/OBSERVABILIDADE_E_MANUTENCAO.md)
- [Relatorio tecnico](docs/RELATORIO_TECNICO.md)
- [Checklist do projeto](docs/CHECKLIST_PROJETO.md)

---

## Estado atual do codigo

Antes de executar o fluxo completo, observe estes pontos do estado atual:

| Area | Estado |
|------|--------|
| Dominio no enum da API | `heart_disease` e `churn` em `schemas/processor_schemas.py` |
| Feature Engineering registrada | Apenas `heart_disease` em `STRATEGY_REGISTRY` |
| Rotas HTTP de treino | `objective` esta tipado como `Literal["churn"]` em `api/v1/endpoints/processor.py` |
| Predicao | `PredictRequest` usa schema estrito de features de `heart_disease` |

Isso significa que a arquitetura ja aponta para multiplos dominios, mas o codigo precisa alinhar `objective` das rotas de treino com `STRATEGY_REGISTRY` para o fluxo HTTP completo funcionar sem ajuste. A documentacao abaixo descreve o funcionamento esperado da aplicacao e registra essas restricoes.

---

## 1. Pre-requisitos

- Docker 24+
- Docker Compose 2.20+
- Portas livres: `8000` (API), `5432` (PostgreSQL), `5050` (pgAdmin), `8080` (Airflow), `8888` (Dozzle)

---

## 2. Configuracao

Crie o arquivo de ambiente:

```bash
cp .env_example .env
```

Preencha, no minimo:

| Variavel | Uso |
|----------|-----|
| `PROJECT_NAME` | Nome exibido pela API |
| `PROJECT_VERSION` | Prefixo das rotas, por exemplo `/v1` |
| `DATABASE_USER`, `DATABASE_PASS`, `DATABASE_NAME` | Credenciais do PostgreSQL |
| `DATABASE_SERVER` | Use `db_processing` no Docker Compose |
| `SECRET` | Chave JWT, gerada com `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` | Configuracao do token |
| `PGADMIN_EMAIL`, `PGADMIN_PASSWORD` | Login do pgAdmin |
| `AIRFLOW_UID` | UID usado pelos containers do Airflow |

Em macOS/Linux, ajuste o UID antes de subir o Compose:

```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

---

## 3. Subir os servicos

```bash
docker compose up --build
```

URLs locais:

| Servico | URL |
|---------|-----|
| Swagger / OpenAPI | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| pgAdmin | http://localhost:5050 |
| Dozzle | http://localhost:8888 |

Parar os containers:

```bash
docker compose down
```

Resetar tambem volumes e banco:

```bash
docker compose down -v
```

---

## 4. Estrutura do projeto

```text
api/v1/endpoints/        Rotas HTTP: auth, users, roles, processor
core/                    Configuracoes, auth, banco, logging e middleware
models/                  Modelos ORM SQLAlchemy
schemas/                 Contratos Pydantic
services/processor/      Orquestracao de treino, promocao e predicao
services/pipelines/      Baseline, Feature Engineering e strategies por dominio
airflow/dags/            DAG de treino
scripts/maintenance/     Relatorios offline de latencia e drift
init_db/                 SQL inicial do banco
docker/                  Dockerfiles
docs/                    Documentacao complementar
```

---

## 5. Autenticacao

O prefixo abaixo usa `/v1`, mas ele vem de `PROJECT_VERSION`.

Criar usuario:

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Usuario Demo",
    "email": "usuario@exemplo.com",
    "password": "senha_segura"
  }'
```

Login:

```bash
curl -X POST http://localhost:8000/v1/auth/authenticate \
  -F "username=usuario@exemplo.com" \
  -F "password=senha_segura"
```

Resposta:

```json
{
  "access_token": "<TOKEN>",
  "token_type": "bearer"
}
```

Use o token em rotas protegidas:

```text
Authorization: Bearer <TOKEN>
```

Novos usuarios entram como perfil comum (`role_id=1`). Rotas administrativas exigem administrador (`role_id=2`).

---

## 6. Contrato do CSV de treino

| Regra | Detalhe |
|-------|---------|
| Formato | CSV com cabecalho |
| Target | A ultima coluna e tratada como alvo |
| Problema | Apenas classificacao binaria |
| ID | Remova colunas de identificador antes do upload |
| Target nulo | Nao permitido |
| Features ausentes | Imputadas no pipeline quando aplicavel |
| Target numerico | Valores `> 0` viram classe positiva `1`; `0` vira classe negativa |

---

## 7. Rotas principais

Todas as rotas abaixo ficam sob `/v1/processor`.

| Metodo | Rota | Auth | Descricao |
|--------|------|------|-----------|
| `POST` | `/predict` | Usuario autenticado | Predicao com modelo ativo |
| `POST` | `/admin/promote` | Admin | Promove um `pipeline_run` concluido |
| `POST` | `/admin/rollback` | Admin | Reativa o deployment arquivado mais recente |
| `GET` | `/admin/deployments/{domain}/history` | Admin | Historico de modelos promovidos |
| `POST` | `/admin/train/baseline` | Admin, nao producao | Treino baseline sincrono |
| `POST` | `/admin/train/feature-engineering` | Admin, nao producao | Treino FE sincrono e retorno de ZIP |
| `POST` | `/admin/train/trigger-dag` | Admin, nao producao | Upload de CSV e disparo do DAG no Airflow |

Em `ENVIRONMENT=prd`, `prod` ou `production`, rotas de treino sincrono e trigger pela API retornam `403`; o treino deve ser feito pelo Airflow.

---

## 8. Fluxo operacional

### 8.1 Treinar baseline

```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/baseline \
  -H "Authorization: Bearer <TOKEN_ADMIN>" \
  -F "file=@dados.csv" \
  -F "objective=churn"
```

A resposta e um CSV. Metadados importantes vem nos headers:

- `X-Pipeline-Run-Id`
- `X-Pipeline-Type`
- `X-Pipeline-Objective`
- `X-Pipeline-Metrics`

### 8.2 Treinar Feature Engineering

```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/feature-engineering \
  -H "Authorization: Bearer <TOKEN_ADMIN>" \
  -F "file=@dados.csv" \
  -F "objective=churn" \
  -F "optimization_metric=recall" \
  -F "time_limit_minutes=2"
```

Metricas aceitas:

```text
accuracy, precision, recall, f1, roc_auc
```

A resposta e um ZIP com artefatos do run e headers `X-Pipeline-*`.

> Ponto de atencao: no codigo atual, `objective=churn` passa pela validacao HTTP, mas o FE exige strategy registrada. Hoje o registry contem `heart_disease`; portanto, para o fluxo FE completo, e necessario alinhar a rota ou implementar/registrar `ChurnFeatures`.

### 8.3 Promover modelo

```bash
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer <TOKEN_ADMIN>" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "heart_disease",
    "pipeline_run_id": 1
  }'
```

A promocao exige:

- `pipeline_run` existente e ativo;
- `status="completed"`;
- `objective` igual ao `domain`;
- arquivo de modelo existente em `model_path`.

### 8.4 Ver historico e rollback

```bash
curl -X GET http://localhost:8000/v1/processor/admin/deployments/heart_disease/history \
  -H "Authorization: Bearer <TOKEN_ADMIN>"
```

```bash
curl -X POST http://localhost:8000/v1/processor/admin/rollback \
  -H "Authorization: Bearer <TOKEN_ADMIN>" \
  -H "Content-Type: application/json" \
  -d '{ "domain": "heart_disease" }'
```

### 8.5 Predicao

O payload atual de predicao esta modelado para `heart_disease` e exige campos ja alinhados ao schema Pydantic:

```bash
curl -X POST http://localhost:8000/v1/processor/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Resposta esperada:

```json
{
  "id": 1,
  "domain": "heart_disease",
  "pipeline_run_id": 3,
  "prediction": 1,
  "probability": 82.0,
  "input_data": {}
}
```

`probability` e retornada em percentual quando o modelo expoe `predict_proba`.

---

## 9. Airflow

O DAG `ml_training_pipeline` executa:

```text
validate_input -> run_baseline -> run_fe -> notify_complete
```

Parametros podem vir de `dag_run.conf` ou da Airflow Variable `ml_training_pipeline_conf`.

Exemplo de JSON:

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

Em producao, use a UI do Airflow para configurar Variables e disparar o DAG manualmente.

---

## 10. Observabilidade e manutencao

| Item | Local |
|------|-------|
| Logs HTTP | `logs/api_requests/access.jsonl` |
| Logs de pipeline | `data/logs/<timestamp>/pipeline_<timestamp>.txt` |
| Relatorios | `artifacts/reports/` |
| MLflow local | `artifacts/mlruns/` |

Relatorio de latencia:

```bash
python scripts/maintenance/latency_report.py --slo-ms 300
```

Relatorio de drift:

```bash
python scripts/maintenance/drift_report.py \
  --train-csv data/referencia.csv \
  --predictions-csv exports/predictions.csv
```

SLO documentado:

| Rota | Metrica | Limite |
|------|---------|--------|
| `POST /processor/predict` | p95 de latencia | `< 300ms` |

---

## 11. Evolucao de dominios

Para adicionar ou completar um dominio:

1. Adicionar o valor em `MLDomain` (`schemas/processor_schemas.py`).
2. Criar uma classe `FeatureStrategy` em `services/pipelines/feature_strategies/`.
3. Registrar a strategy em `STRATEGY_REGISTRY`.
4. Definir labels em `CLASS_LABELS`.
5. Ajustar o schema de predicao para o novo dominio.
6. Garantir que as rotas de treino aceitam o mesmo `objective`.
7. Documentar contrato de CSV, features e payload de predicao.

---

## 12. Limitacoes conhecidas

- Escopo restrito a classificacao binaria tabular.
- FE multi-dominio ainda precisa de alinhamento entre API e registry.
- Predicao esta tipada para features de `heart_disease`.
- Drift e latencia agregada sao analisados por scripts offline, nao por alertas em tempo real.
- MLflow usa tracking local por padrao.
- Treinos sincronizados pela API devem ser usados apenas em desenvolvimento.
