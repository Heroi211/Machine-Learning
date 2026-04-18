# Documentação da solução — Machine Learning Engineering

Documento de referência consolidado: arquitetura, contratos de API, ambientes, orquestração, observabilidade, débitos técnicos e evolução (incluindo **novos domínios de problema**).

**Complementos:** `RELATORIO_TECNICO.md` (EDA, métricas, limitações académicas), `CHECKLIST_PROJETO.md`, `CHECKLIST_QUESTIONAMENTOS.md`, `CHECKLIST_TESTES.md`.

---

## 1. Visão geral

### 1.1 Objetivo do produto

Plataforma para **classificação binária em dados tabulares**, cobrindo:

- Treino (Baseline e Feature Engineering com estratégia por domínio).
- Registo de runs, promoção e rollback de modelos por **domínio** (`heart_disease`, `churn` no enum; churn em evolução no FE).
- **API de predição** autenticada, com payload validado (Pydantic).
- Orquestração opcional via **Apache Airflow** (DAG `ml_training_pipeline`).
- Manutenção offline: relatórios de **latência** e **drift (PSI)**.

### 1.2 Stack principal

| Área | Tecnologia |
|------|------------|
| API | FastAPI, Pydantic v2, JWT |
| Dados | PostgreSQL (async SQLAlchemy), CSV |
| ML | scikit-learn, MLflow (artefatos locais) |
| Orquestração | Airflow 2.x (DAG manual) |
| Infra | Docker Compose |

### 1.3 Prefixo da API

Todas as rotas HTTP ficam sob **`{PROJECT_VERSION}`** (ex.: `/v1`), configurado em `core/configs.py` → `settings.project_version`. Nos exemplos abaixo usa-se **`/v1`** como placeholder.

---

## 2. Arquitetura em camadas

| Camada | Localização | Função |
|--------|-------------|--------|
| Endpoints | `api/v1/endpoints/` | HTTP, autenticação, dependências de ambiente |
| Schemas | `schemas/` | Contratos de entrada/saída, enums (`MLDomain`) |
| Serviços | `services/processor/` | Treino síncrono, predição, deployments |
| Pipelines | `services/pipelines/` | Baseline, FE, `STRATEGY_REGISTRY` por domínio |
| Modelos ORM | `models/` | `PipelineRuns`, `DeployedModels`, `Predictions`, etc. |
| Config | `core/configs.py` | `Settings`, `ENVIRONMENT`, `is_production` |
| Dependências | `core/deps.py` | Sessão DB, utilizador, admin, gates dev/prod |

### 2.1 Fluxo lógico (alto nível)

```
CSV → [Baseline / FE] → pipeline_runs (métricas, modelo)
      → POST /admin/promote → deployed_models (active por domínio)
      → POST /predict → predictions

Manutenção: scripts/maintenance/*.py → artifacts/reports/
```

---

## 3. Ambientes: desenvolvimento vs produção

Configuração via **`ENVIRONMENT`** (ex.: `development`, `staging`, `prd`, `production`, `prod`). A propriedade **`settings.is_production`** considera produção quando o valor normalizado está em `{ prd, prod, production }`.

### 3.1 Comportamento por ambiente

| Funcionalidade | Não produção | Produção |
|----------------|--------------|----------|
| `POST .../admin/train/baseline` | Permitido (admin + `require_sync_training_routes_enabled`) | **403** — usar Airflow |
| `POST .../admin/train/feature-engineering` | Idem | **403** |
| `POST .../admin/train/trigger-dag` | Permitido (admin + `require_airflow_api_trigger_enabled`) | **403** — disparar DAG só pela **UI do Airflow** |
| `POST .../predict` | Permitido (utilizador autenticado) | **Permitido** — serving em produção |

### 3.2 Produção: treino sem API de trigger

1. Colocar o **CSV** num caminho visível pelo worker Airflow (volume partilhado, ex. `ml_shared/...`).
2. **Admin → Variables** no Airflow: configurar **`ml_training_pipeline_conf`** (JSON) e/ou **`ml_training_objective`**, **`ml_training_csv_path`** (ver docstring em `airflow/dags/ml_training_pipeline.py`).
3. **Trigger DAG** na UI (conf vazio `{}` ou overrides pontuais). O DAG faz merge: **defaults das Variables** + **`dag_run.conf`**.

### 3.3 Desenvolvimento

- Treino **síncrono** (baseline/FE) para debug linha a linha.
- **Trigger do DAG pela API** com upload de ficheiro e `conf` gerado automaticamente.

---

## 4. Contratos da API (processor)

Todas as rotas `.../processor/...` exigem **Bearer JWT** salvo onde indicado. Rotas `/processor/admin/*` exigem **administrador** (`role_id` de administrador).

### 4.1 Resumo das rotas

| Método | Rota (após `/v1`) | Auth | Descrição |
|--------|-------------------|------|-----------|
| POST | `/processor/predict` | Utilizador | Body JSON `PredictRequest`: `domain` (`MLDomain`), `features` (schema estrito para `heart_disease`). |
| POST | `/processor/admin/promote` | Admin | JSON: `domain`, `pipeline_run_id`. |
| POST | `/processor/admin/rollback` | Admin | JSON: `domain`. |
| GET | `/processor/admin/deployments/{domain}/history` | Admin | Histórico de deployments do domínio. |
| POST | `/processor/admin/train/baseline` | Admin + não prod | Multipart: `file`, `objective` (enum). |
| POST | `/processor/admin/train/feature-engineering` | Admin + não prod | Multipart + parâmetros de tuning. |
| POST | `/processor/admin/train/trigger-dag` | Admin + não prod | Multipart: CSV + forms; dispara Airflow REST. |

### 4.2 Predição (`POST /processor/predict`)

- **Request:** `domain` + `features` alinhados ao modelo treinado (para `heart_disease`, campos numéricos/booleanos e one-hot com aliases para chaves com espaço/hífen no JSON).
- **Validação:** `extra="forbid"` nos modelos Pydantic — chaves extra são rejeitadas.
- **Resposta:** `prediction`, `probability` (percentagem 0–100 quando disponível), `input_data`, `pipeline_run_id`, etc.
- **Erros típicos:** `404` se não existir deployment ativo para o domínio; `422` se o JSON não cumprir o schema.

### 4.3 Promoção

- Exige run com **`status == completed`**, **`PipelineRuns.active == True`**, ficheiro de modelo existente, e `objective` coerente com o `domain`.
- Em falha de treino síncrono, **`active`** é posto a **`False`** e o run não é elegível.

### 4.4 OpenAPI / Swagger

- **`MLDomain`**: enum exposto (`heart_disease`, `churn`) para forms e bodies onde aplicável.
- Documentação interativa: `{BASE_URL}/docs` (ex.: `http://localhost:8000/docs`).

---

## 5. Pipelines de ML

### 5.1 Baseline

- **Contrato CSV:** última coluna = **`target`**, sem coluna de ID (convenção do projeto).
- Apenas **classificação binária**.
- EDA, imputação, encoding, regressão logística, métricas de teste, MLflow.

### 5.2 Feature Engineering

- Entrada: CSV já no formato esperado pela **estratégia do domínio** (após baseline ou pré-processamento).
- **`STRATEGY_REGISTRY`**: mapeia `objective` → classe `FeatureStrategy` (ex.: `heart_disease` → `HeartDiseaseFeatures`).
- Tuning com limite de tempo e métrica configurável.

### 5.3 Registo de runs

- Tabela **`pipeline_runs`**: tipo de pipeline, objective, status, métricas, caminhos de artefatos, `error_message` em falhas.
- **Nota:** o DAG Airflow **ainda não persiste** linhas em `pipeline_runs` — apenas as rotas síncronas da API criam esses registos. Alinhar DAG ↔ BD é um débito conhecido.

---

## 6. Airflow

- **DAG:** `ml_training_pipeline` — `validate_input` → `run_baseline` → `run_fe` → `notify_complete`.
- **Parâmetros:** merge de **Airflow Variables** + **`dag_run.conf`** (conf do trigger tem prioridade).
- **Logs:** detalhe por task na UI do Airflow; Dozzle mostra sobretudo stdout dos contentores — ver secção 8.

---

## 7. Gestão de modelos e domínios

- **`deployed_models`:** no máximo um registo **active** por domínio (lógica no serviço de deployment).
- **Histórico** e **rollback** permitem comparar versões e reverter para o deployment archived mais recente.

---

## 8. Observabilidade

| Instrumento | Conteúdo |
|-------------|----------|
| `logs/api_requests/access.jsonl` | Latência, método, path, status (middleware configurável). |
| `scripts/maintenance/latency_report.py` | Agregações e SLO (ex.: p95 de `/predict`). |
| `scripts/maintenance/drift_report.py` | PSI entre referência e produção. |
| Dozzle | Logs por contentor (porta típica `8888` no Compose). |
| Airflow UI | Logs por task do DAG. |

**Limitação:** logs finos das tasks ML podem ir para ficheiros no worker; nem tudo aparece no Dozzle sem configurar logging para stdout no Airflow.

---

## 9. Débitos técnicos (estado atual)

| Débito | Impacto | Notas |
|--------|---------|--------|
| DAG não grava `PipelineRuns` | Treino “oficial” só Airflow não alimenta a mesma tabela que a API usa para promote | Persistir runs nas tasks ou job assíncrono. |
| `churn` no enum sem strategy FE completa | FE com `churn` falha até haver classe em `STRATEGY_REGISTRY` | Completar `ChurnFeatures` + testes. |
| Critérios de promoção no relatório vs código | Documentação fala em Δmétrica ≥ 2% e PSI; promoção valida run completo e ficheiros | Automatizar gates de negócio se necessário. |
| `load_dotenv()` disperso vs `Settings` | Possível redundância em módulos legados | Centralizar env (ver checklist de questionamentos). |
| MLflow em SQLite local | Adequado a dev; concorrência limitada em equipa | Servidor MLflow partilhado em produção. |
| Checklist projeto: itens P1/P2 abertos | Ex.: listagem de runs, `reason` em promote | Ver `CHECKLIST_PROJETO.md`. |

---

## 10. Melhorias futuras sugeridas

1. **Persistência unificada:** qualquer caminho de treino (API síncrona ou DAG) gera `pipeline_runs` coerente e IDs rastreáveis até ao promote.
2. **Predict multi-domínio:** `Union` discriminada por `domain` ou endpoints por domínio; feature store ou contratos versionados.
3. **Filas de trabalho:** Celery/RQ para treinos longos sem bloquear API (se voltar a expor treino pesado via HTTP).
4. **Observabilidade:** exportar métricas (Prometheus), traces; alinhar logs ML ao stdout em prd para agregadores.
5. **Segurança:** rate limit em `/predict`, auditoria de promote/rollback com campo `reason`.
6. **Testes automatizados:** CI com subset do `CHECKLIST_TESTES.md`.

---

## 11. Incluir um novo domínio de problema

Objetivo: suportar um novo `objective` (ex.: `fraud`, `credit_risk`) mantendo o padrão **binário tabular** e a arquitetura atual.

### 11.1 Checklist técnico

1. **`MLDomain` (`schemas/processor_schemas.py`)**  
   - Adicionar membro ao enum (ex.: `fraud = "fraud"`).  
   - O Swagger passa a listar o valor automaticamente.

2. **Feature Engineering — `STRATEGY_REGISTRY` (`services/pipelines/feature_strategies/`)**  
   - Implementar classe `FeatureStrategy` com `validate`, `build`, colunas esperadas.  
   - Registar: `"fraud": FraudFeatures` (nome do ficheiro/classe ao critério do projeto).

3. **Labels (`get_class_labels` / `CLASS_LABELS`)**  
   - Definir tuplo (negativo, positivo) para gráficos e Baseline.

4. **Baseline**  
   - Garantir CSV respeita convenção (última coluna `target`).  
   - Dataset de treino e pré-processamento alinhados ao domínio.

5. **Predição (`POST /predict`)**  
   - Hoje o body é centrado em `HeartDiseaseFeaturesInput`. Para outro domínio é necessário **um destes desenhos**:  
   - **A)** Modelo Pydantic novo (ex.: `FraudFeaturesInput`) + **`Union`** discriminada por `domain` em `PredictRequest`; ou  
   - **B)** Endpoint separado `POST /processor/predict/fraud`; ou  
   - **C)** `features` como JSON validado por schema dinâmico (menos tipagem no Swagger).  
   - Ajustar `processor_service.predict_for_domain` / `_prepare_prediction_features` se a estratégia de pré-processamento for diferente.

6. **Airflow / Variables**  
   - O DAG já usa `objective` no `conf`; após registo no registry, `validate_input` aceita o novo domínio se o CSV tiver colunas validadas pela strategy.

7. **Testes manuais**  
   - Fluxo: baseline (dev) ou DAG → FE → promote → predict com deployment ativo.

8. **Documentação e dados**  
   - Atualizar README / relatório com origem do dataset e limitações do domínio.

### 11.2 Armadilhas comuns

- Esquecer o enum **`MLDomain`** → cliente pode enviar strings válidas na BD mas rejeitadas na API.  
- **FE** sem strategy → `ValueError` na rota síncrona.  
- **Predict:** nomes de features do modelo (`feature_names_in_`) devem bater com o que o pré-processamento gera (ordem/alinhamento em `processor_service`).

---

## 12. Referência rápida de URLs (desenvolvimento local)

| Serviço | URL típica |
|---------|------------|
| Swagger | http://localhost:8000/docs |
| Airflow | http://localhost:8080 |
| pgAdmin | http://localhost:5050 |
| Dozzle | http://localhost:8888 |

Credenciais e portas podem variar — ver `.env` e `docker-compose.yaml`.

---

## 13. Manutenção deste documento

- Alterações de produto (gates por ambiente, novos endpoints) devem refletir-se aqui e nos checklists.  
- O **`RELATORIO_TECNICO.md`** mantém o foco em EDA, métricas e entrega académica; este ficheiro é a **visão operacional e de integração** da solução.

*Última consolidação: alinhada ao código e aos checklists do repositório (ambientes, MLDomain, DAG com Variables, `PipelineRuns`, contratos Pydantic).*
