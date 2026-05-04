# Tech Challenge — Previsão de Churn com Rede Neural (MLP / PyTorch)

Pipeline End-to-End de Machine Learning para classificação binária de **churn** em telecomunicações: ingestão → baseline → feature engineering → MLP PyTorch → API de inferência (FastAPI) → orquestração (Airflow) → tracking (MLflow) → monitoramento (drift / latência).

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Escopo e Limitações](#2-escopo-e-limitações)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Fluxos de Execução](#4-fluxos-de-execução)
   - [4.1 Fluxo Manual (API)](#41-fluxo-manual-api)
   - [4.2 Fluxo Automatizado (Airflow)](#42-fluxo-automatizado-airflow)
5. [Engenharia de Features](#5-engenharia-de-features)
6. [Modelo de Machine Learning](#6-modelo-de-machine-learning)
7. [Regras de Decisão](#7-regras-de-decisão)
8. [Entradas e Saídas](#8-entradas-e-saídas)
9. [Como Executar o Projeto](#9-como-executar-o-projeto)
10. [Melhorias Futuras](#10-melhorias-futuras)

---

## 1. Visão Geral

### Problema de negócio
Operadora de telecomunicações com cancelamento (churn) acelerado precisa de um modelo preditivo que classifique clientes com risco de cancelamento, alimentando ações de retenção da equipe de CRM.

### Objetivo do sistema
- Treinar e versionar uma **rede neural (MLP, PyTorch)** para prever P(churn) por cliente.
- Comparar o MLP com baselines lineares e árvores (Scikit-Learn) sob a mesma metodologia (`StratifiedKFold` + holdout estratificado).
- Servir a inferência via **API FastAPI** autenticada (JWT), com rastreio em **MLflow** e orquestração em **Airflow**.

### Contexto do desafio (PDF `reference/TC_01.pdf`)
- Tema central: **Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End**.
- Dataset sugerido: **Telco Customer Churn (IBM)** — variáveis tabulares, classificação binária, ≥ 5.000 registros.
- Bibliotecas obrigatórias: **PyTorch**, **Scikit-Learn**, **MLflow**, **FastAPI**.
- Boas práticas exigidas: seeds fixos, validação cruzada estratificada, Model Card, testes (smoke, schema, API), logging estruturado, **ruff** sem erros, `pyproject.toml` como single source of truth.
- Etapas: (1) EDA + ML Canvas + baselines, (2) MLP + comparação + custo, (3) refatoração + API + testes + Makefile, (4) Model Card + README + vídeo STAR.

---

## 2. Escopo e Limitações

### O que o sistema FAZ
- Ingere CSV tabular e executa **pipeline Baseline** (`LogisticRegression` com `class_weight='balanced'`) com pré-processamento aprendido pós-split (sem leakage).
- Executa **pipeline de Feature Engineering** com criação de features de domínio (`ChurnFeatures`), comparação cruzada de 4 modelos sklearn (Decision Tree, Random Forest, SVM, Gradient Boosting), tuning por `ParameterSampler` com time-budget e guardrails (`min_precision`, `min_roc_auc`).
- Treina, **em paralelo no mesmo run FE**, uma **MLP em PyTorch** com early stopping pela `val_loss` (`BCEWithLogitsLoss`, `AdamW`).
- Persiste artefatos: `joblib` do pipeline sklearn, bundle MLP (`.pt` + `_preprocess.joblib` + `_meta.json`), CSVs pré/pós-transform, gráficos, `manifest.json`.
- Versiona experimentos no **MLflow** (params, métricas, dataset CSV, gráficos).
- Promove o modelo campeão por domínio (`POST /admin/promote`) e faz **rollback** sob demanda.
- Serve **`POST /predict`** autenticado, retornando `prediction` (0/1) e `probability`.
- Orquestra o fluxo Baseline → FE → (auto)promote via **Airflow** (`ml_training_pipeline`).
- Provê scripts offline de **drift (PSI)** e **latência (p50/p90/p95/p99)**.

### O que o sistema NÃO FAZ
- **Não** suporta regressão nem classificação multiclasse — apenas **classificação binária**.
- **Não** estima valor monetário de churn (LTV) — apenas P(churn).
- **Não** explica causalidade — usa importância (Gini / permutação) como proxy.
- **Não** monitora drift em tempo real — relatório PSI é offline / agendado manualmente.
- **Não** aplica reamostragem (SMOTE, undersampling). O desbalanceamento é mitigado por `class_weight='balanced'` no baseline e otimização orientada a `recall`.
- **Não** hospeda MLflow remoto por padrão — tracking é via SQLite local em volume compartilhado.
- **Não** valida deriva de schema em produção — categorias novas são absorvidas pelo `OneHotEncoder(handle_unknown="ignore")` (silenciosamente).

### Limitações técnicas
- **Dataset**: cohort estático do Telco (~7.043 linhas, EUA). Não captura sazonalidade recente, mudanças regulatórias ou outros mercados.
- **Modelo MLP**: arquitetura fixa por padrão (`Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear(1)`); hidden_dims `(64, 32)`. Não há HPO automatizado da MLP — apenas dos modelos sklearn.
- **Servidor de inferência**: o `inference_backend` é decidido no **treino** (env `USE_MLP_FOR_PREDICTION`); alternar a env afeta apenas novos runs/promotes, não muda o que já está em produção.
- **Treino síncrono via API**: roda na thread HTTP. Em ambiente `production` (env `ENVIRONMENT=prod|production`), as rotas síncronas são **desabilitadas** — somente Airflow.
- **Cross-container**: caminhos persistidos pelo Airflow são remapeados em runtime (`/opt/airflow/ml_project/` ↔ `/var/www/ml_shared/`).
- **Domínios**: além de `churn` (foco do desafio), o sistema mantém o domínio legado `heart_disease` para validar a generalidade da arquitetura.

### Suposições assumidas
- Custo de **FN >> FP** em telecom (5–10×): perder cliente que iria cancelar é mais caro que oferecer retenção a quem ficaria. Por isso a métrica de otimização padrão é **`recall`**.
- O CSV de treino chega no contrato: **última coluna = target**, **sem coluna de ID**, sem multiclasse.

---

## 3. Arquitetura do Sistema

### Componentes

| Componente | Responsabilidade | Localização |
|-----------|-----------------|-------------|
| **API FastAPI** | Autenticação JWT, `/predict`, rotas admin (treino, runs, promote, rollback, history) | `src/api/v1/` + `main.py` |
| **Pipeline Baseline** | EDA leve, qualidade, target, split, `LogisticRegression`, manifest + sample raw | `src/services/pipelines/baseline.py` |
| **Pipeline Feature Engineering** | Strategy de features, comparação de 4 modelos, tuning com guardrails, **treino paralelo da MLP PyTorch**, exporte de bundle, MLflow | `src/services/pipelines/feature_engineering.py` |
| **MLP PyTorch** | `_MLPBinary` + `train_eval_mlp_binary_tabular` (early stopping, BCE com logits) | `src/services/pipelines/mlp_torch_tabular.py` |
| **Inferência MLP** | Carrega bundle (`.pt` + preprocess + meta), aplica `ColumnTransformer` e prediz | `src/services/pipelines/mlp_inference.py` |
| **Feature Strategies** | Lógica de features por domínio (`ChurnFeatures`, `HeartDiseaseFeatures`) | `src/services/pipelines/feature_strategies/` |
| **Processor / Deployment** | Persistência de runs, eleição de campeão (`cv_recall`), promote, rollback | `src/services/processor/` |
| **PostgreSQL** | `users`, `roles`, `pipeline_runs`, `deployed_models`, `predictions` | `init_db/database.sql` |
| **MLflow** | Tracking (params, metrics, artefatos, dataset CSV) — backend SQLite | `src/artifacts/mlruns/` (volume `ml_shared`) |
| **Airflow** | DAG `ml_training_pipeline` (LocalExecutor) — Baseline → FE → promote opcional | `airflow/dags/ml_training_pipeline.py` |
| **Scripts de manutenção** | Relatório de drift (PSI) e latência (SLO p95) | `src/scripts/maintenance/` |

### Fluxo de dados ponta a ponta

```text
CSV bruto (host)
    └─> ml_data/uploads/  (bind mount)
            ├─> /var/www/ml_shared/uploads/   (API)
            └─> /opt/airflow/ml_project/uploads/  (Airflow)

[Baseline]
    csv → load_data → missing/outliers → target → split →
    ColumnTransformer (median+scaler / passthrough / mode+OHE) → LogisticRegression →
    artefatos: baseline_sample.csv (raw_clean) + manifest.json + joblib
    └─> snapshot:  src/data/logs/<ts>/   |  pre_processed/ (se vencedor por test_recall)
    └─> MLflow:    experiment "<obj>_baseline"

[Feature Engineering]
    manifest.json → carrega baseline_sample.csv → strategy.build (ChurnFeatures) →
    split estratificado → 4 modelos sklearn (CV 5-fold, optimization_metric) →
    seleção por cv_<metric> com guardrails → tuning ParameterSampler (time-budget) →
    treino paralelo da MLP PyTorch (mesmo ColumnTransformer, sem SelectKBest) →
    artefatos: best_<obj>_<ts>.joblib + bundle MLP (.pt + _preprocess.joblib + _meta.json)
    └─> ZIP fe_artifacts_<run_id>_<ts>.zip (resposta da API)
    └─> MLflow: experiment "<obj>_feature_engineering"

[Promote]
    pipeline_runs (active=true, status=completed, FE) → DeployedModels (status=active)
    Antigo "active" vira "archived". Constraint UNIQUE garante 1 ativo por domínio.

[Predict]
    POST /predict → resolve deployment ativo do domínio →
    inference_backend == "mlp" ? carrega bundle MLP (PyTorch) : joblib.load (sklearn) →
    aplica strategy.build / clean → predict → predict_proba → grava em predictions
```

### Conexões entre módulos

- **Contrato Baseline → FE**: `manifest.json` (`output_sample_csv_stable`, `sample_schema = "raw_clean"`, `model_path`, `feature_groups`).
- **Decisão de inference_backend**: lida da env `USE_MLP_FOR_PREDICTION` no momento do **treino FE** e gravada em `pipeline_runs.inference_backend`. O `/predict` lê do run, não da env.
- **Volume `ml_shared`**: API e Airflow compartilham `/var/www/ml_shared` ↔ `/opt/airflow/ml_project`. Caminhos cross-container são remapeados em `_resolve_shared_artifact_path`.
- **Banco `pipeline_runs`**: separa runs manuais (`is_airflow_run=false`) de Airflow (`true`); o comparador de campeão é estritamente intra-grupo.

---

## 4. Fluxos de Execução

### 4.1 Fluxo Manual (API)

Sequência completa via `curl` (alternativa: Swagger em `/docs`).

#### Pré-condições
- `docker compose up --build` em execução.
- CSV de treino disponível (ex.: `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- Variáveis `OBJECTIVE=churn`, `USE_MLP_FOR_PREDICTION` definidas em `.env`.

#### 1. Criar conta + autenticar
```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Admin","email":"admin@exemplo.com","password":"senha"}'

curl -X POST http://localhost:8000/v1/auth/authenticate \
  -F "username=admin@exemplo.com" -F "password=senha"
# → {"access_token":"<TOKEN>","token_type":"bearer"}
```
> Para promover/treinar é necessário `role_id = 2` (Administrator). O seed (`init_db/database.sql`) cria 2 usuários admin.

#### 2. Treino Baseline (admin)
```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/baseline \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@./WA_Fn-UseC_-Telco-Customer-Churn.csv"
```
- **Saída**: CSV pré-processado (`baseline_sample.csv`, `sample_schema=raw_clean`) no corpo + cabeçalhos `X-Pipeline-Run-Id`, `X-Pipeline-Metrics`.
- **Persistência**: `pipeline_runs(pipeline_type='baseline', status='completed')` com `model_path` apontando para o `.joblib`.

#### 3. Treino Feature Engineering (admin)
```bash
curl -X POST http://localhost:8000/v1/processor/admin/train/feature-engineering \
  -H "Authorization: Bearer <TOKEN>" \
  -F "optimization_metric=recall" \
  -F "time_limit_minutes=2"
```
- **Entrada**: nenhuma (lê `manifest.json` do baseline ativo).
- **Saída**: ZIP (`fe_artifacts_<run_id>_<ts>.zip`) com `best_<obj>_<ts>.joblib`, bundle MLP, `model_comparison_full.md`, CSVs pré/pós-transform, gráficos.
- **Cabeçalhos**: `X-Pipeline-Run-Id`, `X-Pipeline-Metrics` (inclui `cv_recall`, `best_model_name`, `inference_backend`).

#### 4. Listar runs (admin)
```bash
curl -X GET "http://localhost:8000/v1/processor/admin/runs?pipeline_type=feature_engineering&status=completed&limit=20" \
  -H "Authorization: Bearer <TOKEN>"
```

#### 5. Promover modelo (admin)
```bash
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer <TOKEN>"
```
- Promove o **único** run FE `active=true`, `status=completed`, com `inference_backend` coerente com `USE_MLP_FOR_PREDICTION`.
- Antigo deployment `active` → `archived`.

#### 6. Predição (qualquer usuário autenticado)
```bash
curl -X POST http://localhost:8000/v1/processor/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "churn",
    "features": {
      "gender": "Female", "seniorcitizen": 0, "partner": 1, "dependents": 0,
      "tenure": 12, "phoneservice": 1, "multiplelines": 0,
      "internetservice": "Fiber optic", "onlinesecurity": 0, "onlinebackup": 0,
      "deviceprotection": 0, "techsupport": 0, "streamingtv": 1, "streamingmovies": 1,
      "contract": "Month-to-month", "paperlessbilling": 1,
      "paymentmethod": "Electronic check", "monthlycharges": 95.5, "totalcharges": 1146.0
    }
  }'
```
- **Saída**: `{ "id": <int>, "domain": "churn", "pipeline_run_id": <int>, "prediction": 0|1, "probability": <0..100>, "input_data": {...} }`.

#### 7. Rollback (admin, opcional)
```bash
curl -X POST http://localhost:8000/v1/processor/admin/rollback -H "Authorization: Bearer <TOKEN>"
```
Reativa o último deployment `archived` do domínio.

### 4.2 Fluxo Automatizado (Airflow)

#### Ferramenta
- **Apache Airflow** (LocalExecutor) — `airflow_scheduler` + `airflow_webserver` (UI em `http://localhost:8080`, user/pass: `airflow`/`airflow`).
- DAG: **`ml_training_pipeline`** (`airflow/dags/ml_training_pipeline.py`).

#### Configuração

A configuração efetiva é `merge(Airflow Variable, dag_run.conf)` — conf do trigger sobrescreve a Variable.

Variable `ml_training_pipeline_conf` (JSON, bootstrap em `airflow/bootstrap/ml_training_pipeline_conf.json`):
```json
{
  "objective": "churn",
  "csv_path": "/opt/airflow/ml_project/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv",
  "optimization_metric": "recall",
  "time_limit_minutes": 30,
  "tuning_n_iter": 1000,
  "user_id": 2,
  "auto_promote": false
}
```
Campos opcionais: `min_precision`, `min_roc_auc`, `acc_target`, `decision_threshold`.

#### Tasks

| Ordem | Task | Descrição |
|-------|------|-----------|
| 1 | `validate_input` | Confirma CSV e domínio (`STRATEGY_REGISTRY[obj].validate`). Empilha XComs. |
| 2 | `deactivate_manual_runs` | Desativa em BD runs **manuais** (`is_airflow_run=false`) do `objective`. |
| 3 | `run_baseline` | Executa Baseline com `defer_global_preprocess_contract=True`; persiste `pipeline_runs` com `is_airflow_run=true`. |
| 4 | `run_fe` | Executa FE; comparador `cv_<metric>` decide `active`; persiste `fe_pipeline_run_id` e `fe_recall_champion`. |
| 5 | `promote_fe_optional` | Se `auto_promote=true` **e** o run venceu o comparador → chama o mesmo fluxo do `/admin/promote`. |
| 6 | `notify_complete` | Loga IDs e métricas para decisão. |

#### Gatilhos
- **Manual via UI**: `Trigger DAG` em `http://localhost:8080/dags/ml_training_pipeline/grid`.
- **Manual via CLI**: `airflow dags trigger ml_training_pipeline --conf '{...}'`.
- **Manual via API REST do Airflow** (já usado pela rota da API):
  ```bash
  curl -X POST http://localhost:8000/v1/processor/admin/train/trigger-dag \
    -H "Authorization: Bearer <TOKEN>" \
    -F "file=@./WA_Fn-UseC_-Telco-Customer-Churn.csv" \
    -F "optimization_metric=recall" \
    -F "time_limit_minutes=30"
  ```
- **Schedule**: `schedule_interval=None` (apenas trigger explícito — não há cron por padrão).

---

## 5. Engenharia de Features

### Strategy aplicada (`ChurnFeatures` — `src/services/pipelines/feature_strategies/churn_features.py`)

#### Colunas obrigatórias
`tenure`, `monthlycharges`, `totalcharges`, `contract`, `paymentmethod`, `internetservice`.

#### Features criadas

| Feature | Definição | Racional |
|---------|-----------|----------|
| `is_new_customer` | `tenure <= 12` | Clientes novos têm taxa de churn historicamente maior. |
| `tenure_log` | `log1p(tenure)` | Estabiliza variância e suaviza assimetria. |
| `contract_stability` | Mapeia `Month-to-month`/`One year`/`Two year` → 0/1/2 | Capta lock-in contratual; ordinal direto. |
| `new_customer_in_mounth_contract` | Cruza `is_new_customer` & `Month-to-month` | Combina dois fatores de risco. |
| `risk_payment_monthly` | `Electronic check` & `Month-to-month` | Padrão histórico de inadimplência/saída. |
| `new_customer_risk_payment_monthly` | `risk_payment_monthly` & `is_new_customer` | Tripla intersecção de risco. |
| `fiber_high_cost` | `Fiber optic` & `monthlycharges > median` | Clientes de fibra com fatura alta — perfil de churn mapeado. |
| `fiber_premium_monthly` | `Fiber optic` & `Month-to-month` | Sem lock-in + serviço caro. |
| `fiber_premium_monthly_new_customer` | Anterior & `is_new_customer` | Tripla intersecção. |
| `avg_ticket` | `totalcharges / (tenure + 1)` | Ticket médio do cliente; suaviza para clientes de tenure 0. |
| `charge_ratio` | `monthlycharges / (avg_ticket + ε)` | Detecta upgrade/downgrade recente. |
| `num_services` | Soma de 6 serviços (security/backup/protection/support/tv/movies) | Engagement com o portfólio. |
| `low_engagement` | `num_services <= 2` | Baixo engagement → churn. |
| `high_cost_low_engagement` | `monthlycharges > median` & `num_services <= 2` | Cliente caro com pouca ancoragem ao produto. |
| `is_auto_payment` | `paymentmethod` contém `automatic` | Pagamento automático reduz fricção e churn. |

#### Estado persistido na strategy
- `monthly_median` (mediana de `monthlycharges` no fit) é guardada em `pipeline_runs.metrics["strategy_monthly_charges_median"]` e **reidratada na inferência** para garantir consistência treino/serving.

### Pré-processamento pós-strategy (igual no FE e na inferência)
- **Numéricas contínuas**: `SimpleImputer(median)` (apenas se houver nulos) + `StandardScaler`.
- **Numéricas binárias** (subset detectado): `passthrough`.
- **Categóricas**: `SimpleImputer(most_frequent)` (se houver nulos) + `OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)`.
- **VarianceThreshold(0.0)** + **`SelectKBest(f_classif, k=25)`** (apenas no SkPipeline; **não** no MLP).

### Riscos técnicos

| Risco | Estado |
|-------|--------|
| **Leakage** | Mitigado: imputação/encoding aprendidos **só no treino** (após `train_test_split`), via `ColumnTransformer` dentro do `SkPipeline`. Mediana de `monthlycharges` no fit é reusada na inferência. |
| **Drift de categorias** | `OneHotEncoder(handle_unknown="ignore")` evita falha mas pode degradar silenciosamente — exige PSI. |
| **Vieses demográficos** | Não há análise por subgrupo (idade, dependentes, gênero) na entrega atual — dívida explícita no Model Card. |
| **Selection bias** | Dataset cobre apenas clientes que já contrataram; clientes que cancelaram pre-onboarding não estão representados. |
| **Cohort temporal** | Snapshot histórico — performance pode degradar com sazonalidade ou mudanças de mercado. |

---

## 6. Modelo de Machine Learning

### Tipo de modelo

#### Modelo principal — **MLP PyTorch** (foco do desafio)
- **Arquitetura**: `Linear(n, 64) → ReLU → Dropout → Linear(64, 32) → ReLU → Dropout → Linear(32, 1)` (1 logit).
- **Loss**: `BCEWithLogitsLoss`.
- **Otimizador**: `AdamW(lr=1e-3, weight_decay=1e-5)`.
- **Batch**: `DataLoader(batch_size=64, shuffle=True)`.
- **Early stopping**: `patience=20` épocas sobre `val_loss` (split estratificado de 15% do treino).
- **Max épocas**: 300.
- **Pré-processamento servido**: o `ColumnTransformer` (sem `SelectKBest`) é serializado em `*_preprocess.joblib` no bundle.

#### Baselines
- **Logistic Regression** (`class_weight='balanced'`, `max_iter=1000`) — modelo do pipeline `Baseline`.
- **Decision Tree, Random Forest, SVM (RBF), Gradient Boosting** — comparados no FE.

### Justificativa
- **MLP**: requisito do desafio; capta interações não-lineares entre features de domínio sem engineering manual exaustivo.
- **Logistic Regression como baseline**: linha de base interpretável e barata, exigida pela Etapa 1 do PDF.
- **Random Forest / Gradient Boosting**: alto desempenho em tabular, comparação obrigatória (Etapa 2).
- **Decisão de qual servir**: controlada pela env `USE_MLP_FOR_PREDICTION` (gravada no `pipeline_runs.inference_backend` no momento do treino). O `/predict` honra esse campo do run promovido — não a env atual.

### Como foi treinado

1. **Split estratificado** 80/20 com `random_state=42`.
2. **CV `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** sobre o treino.
3. **Comparação de modelos sklearn** com `cross_val_score(scoring=optimization_metric)` + `cross_validate` para guardrails (`cv_precision`, `cv_roc_auc`).
4. **Seleção** do `best_model_name` por maior `cv_<metric>` entre os que passam guardrails (fallback: maior `cv_<metric>` sem guardrails, com aviso).
5. **Tuning** via `ParameterSampler` com **time-budget** (`time_limit_minutes`) e `tuning_n_iter` (default 100). Guardrails reaplicados.
6. **MLP PyTorch** treinada em paralelo no mesmo run, sobre o mesmo `ColumnTransformer.fit_transform(x_train)` (sem `SelectKBest`), com seu próprio split treino/val (15%).
7. **MLflow**: cada run abre um `mlflow.start_run()` em experimentos `<obj>_baseline` e `<obj>_feature_engineering`; loga params, métricas (treino/val/teste), gráficos (PR-curve, importância) e dataset CSV.
8. **Persistência** em `pipeline_runs` (Postgres) + bundle no volume `ml_shared`.

### Métricas utilizadas

| Métrica | Onde |
|---------|------|
| `accuracy`, `precision`, `recall`, `f1` | Treino + teste, todos os modelos. |
| `roc_auc` | Teste (todos os modelos com `predict_proba`/`decision_function`). |
| `pr_auc` (`average_precision`) | Treino + teste do baseline. |
| `cv_<metric>` | CV 5-fold do treino, métrica que dirige seleção e tuning. |
| `cv_precision`, `cv_roc_auc` | Usadas como guardrails opcionais. |
| `overfitting_gap` | `train_accuracy - test_accuracy` (logado no baseline). |

### Limitações do modelo
- MLP com **arquitetura fixa** — sem HPO automatizada.
- Threshold de decisão é **escalar** — não calibrado por subgrupo.
- O `SelectKBest(k=25)` aplica-se só ao SkPipeline; **MLP vê todas as features pós-OHE**, o que pode desfavorecer comparações em datasets curtos ou ruidosos.
- Sem calibração explícita de probabilidade (Platt / isotônica) — `probability` retornada é a saída direta (sigmoid no MLP, `predict_proba` no sklearn).

---

### 6.1 Implementação detalhada da MLP (PyTorch)

> Subseção didática para leitores sem familiaridade com redes neurais. O que está descrito abaixo é o que **de facto** está no código — não teoria genérica.

#### O que é a MLP em uma frase
Uma **MLP (Multi-Layer Perceptron)** é uma rede neural *feedforward* que recebe um vetor numérico de features e produz **um único número** (o "logit"). Aplicando a função `sigmoid` a esse logit obtém-se a probabilidade de churn (entre 0 e 1).

#### Onde está o código

| Responsabilidade | Arquivo |
|------------------|---------|
| Definição da rede + treino + avaliação | `src/services/pipelines/mlp_torch_tabular.py` |
| Carregamento do bundle + predição em produção | `src/services/pipelines/mlp_inference.py` |
| Integração com o pipeline FE | `FeatureEngineering._run_mlp_torch_mvp()` em `src/services/pipelines/feature_engineering.py` |

#### Arquitetura — camada a camada

Classe `_MLPBinary` (em `mlp_torch_tabular.py`). Defaults: `hidden_dims=(64, 32)`, `dropout=0.0`. Para um vetor de entrada com `N` features (após `OneHotEncoder` + `StandardScaler`):

| # | Camada | Forma | O que faz (em linguagem simples) |
|---|--------|-------|----------------------------------|
| 1 | `Linear(N, 64)` | N → 64 | 64 combinações lineares das features (cada neurônio aprende um peso por feature + bias). |
| 2 | `ReLU()` | 64 → 64 | Não-linearidade: zera valores negativos, mantém positivos. Sem isto, a rede inteira viraria uma única regressão linear. |
| 3 | `Dropout(p)` | 64 → 64 | **Só durante o treino**: desliga aleatoriamente uma fração `p` dos neurônios. Regulariza (evita decorar o treino). Como `p=0` por default, este passo é praticamente identidade. |
| 4 | `Linear(64, 32)` | 64 → 32 | Reduz dimensão e aprende combinações de "alto nível" (ex.: "cliente novo + fibra + caro"). |
| 5 | `ReLU()` | 32 → 32 | Não-linearidade. |
| 6 | `Dropout(p)` | 32 → 32 | Idem item 3. |
| 7 | `Linear(32, 1)` | 32 → 1 | Camada de saída. **Sem ativação aqui** — produz o **logit** (qualquer número real). |

Forward: `model(x).squeeze(-1)` → vetor 1D de logits (um por exemplo).

#### Por que **não** existe `Sigmoid` na última camada?
A `BCEWithLogitsLoss` (loss usada no treino) aplica `sigmoid + binary cross-entropy` numa **única operação numericamente estável**. Aplicar `sigmoid` antes da loss pode causar overflow/underflow em logits extremos. Esta é a forma recomendada pela própria documentação do PyTorch para classificação binária.

#### Loop de treino — passo a passo

Implementado em `train_eval_mlp_binary_tabular`:

1. **Sementes determinísticas**: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)` (se GPU), e um `torch.Generator(seed=42)` separado para o `DataLoader` (garante mesma ordem de batches em cada execução).
2. **Tensores**: `X` convertido para `float32` denso (matrizes esparsas viram `.toarray()`); `y` em `float32` (BCE exige float). Tudo movido para `device` (CUDA se disponível, senão CPU).
3. **DataLoader**: `TensorDataset(X_train, y_train)` com `batch_size=min(64, len(X_train))`, `shuffle=True`, `drop_last=False`.
4. **Para cada época** (até `max_epochs=300`):
   - `model.train()` — ativa Dropout (se houver).
   - **Para cada batch**:
     - `optimizer.zero_grad(set_to_none=True)` — zera gradientes do passo anterior.
     - `logits = model(xb)` — forward pass.
     - `loss = BCEWithLogitsLoss(logits, yb)` — calcula erro (sigmoid + BCE em um só passo).
     - `loss.backward()` — backpropagation (calcula ∂loss/∂peso para todos os parâmetros).
     - `optimizer.step()` — `AdamW` atualiza os pesos (`lr=1e-3`, `weight_decay=1e-5`).
   - **Fim da época**:
     - `model.eval()` (desliga Dropout) + `torch.no_grad()` (não calcula gradientes — economiza memória).
     - Calcula `val_loss` sobre o conjunto inteiro de validação.
5. **Early stopping** (`patience=20`):
   - Se `val_loss < melhor_anterior - 1e-6` → salva `state_dict` atual como `best_state`, zera o contador.
   - Caso contrário → decrementa contador. Se chegar a 0 → para o treino.
6. **Restauração**: ao final, carrega `best_state` (não os pesos da última época). É comum a última época estar pior que o melhor ponto.

#### Por que `AdamW` e não SGD?
`AdamW` é **adaptativo** — ajusta a taxa de aprendizado por parâmetro automaticamente, e separa `weight_decay` da atualização de gradiente (mais correto matematicamente que o `Adam` clássico). É a escolha padrão moderna para tabular pequeno/médio.

#### Avaliação interna (durante o treino)

Após carregar `best_state`:
- `logits_test = model(X_test)` (sem gradientes).
- `proba_test = 1 / (1 + exp(-logits_test))` — `sigmoid` manual em NumPy.
- `pred_test = (proba_test >= 0.5).astype(int64)` — **threshold fixo de 0.5** apenas para reportar métricas no MLflow.
- Métricas calculadas: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (se y_true tiver as duas classes).

> **Atenção**: o `0.5` aqui é **só** para reporte interno (MLflow / `pytorch_mvp_summary.csv`). O threshold de produção é o da env `CLASSIFICATION_DECISION_THRESHOLD`, gravado no `_meta.json` e usado por `predict_with_mlp`.

#### Hiperparâmetros (defaults em `FeatureEngineering.__init__`)

| Parâmetro | Default | O que significa na prática |
|-----------|---------|----------------------------|
| `mlp_hidden_dims` | `(64, 32)` | Duas camadas ocultas — capacidade moderada para o tamanho do Telco. |
| `mlp_dropout` | `0.0` | Sem dropout — dataset relativamente pequeno e limpo. Aumentar (ex.: 0.2) ajuda em datasets ruidosos. |
| `mlp_batch_size` | `64` | Equilíbrio típico para ~7k linhas. |
| `mlp_lr` | `1e-3` | Taxa de aprendizado padrão do `AdamW`. |
| `mlp_weight_decay` | `1e-5` | L2 regularization "leve" embutida. |
| `mlp_max_epochs` | `300` | Teto duro — o early stopping costuma parar muito antes. |
| `mlp_early_stopping_patience` | `20` | Tolera 20 épocas sem melhora antes de parar. |
| `mlp_val_fraction` | `0.15` | 15% do treino reservado para `val_loss` (estratificado). |

#### Pré-processamento da MLP vs sklearn — diferença importante

| Etapa | Pipelines sklearn (DT/RF/SVM/GB) | MLP PyTorch |
|-------|----------------------------------|-------------|
| `ColumnTransformer` (imputer + scaler + OHE) | ✅ | ✅ (mesmo objeto, mesmo `fit_transform` no treino) |
| `VarianceThreshold(0.0)` | ✅ | ❌ |
| `SelectKBest(f_classif, k=25)` | ✅ | ❌ |
| Modelo final | DT/RF/SVM/GB | `_MLPBinary` |

**Por que a MLP não usa `SelectKBest`?**
- A própria primeira camada (`Linear(N, 64)`) aprende a "anular" features irrelevantes via pesos próximos de zero — não precisa de seleção prévia.
- Forçaria a MLP a competir em desigualdade com sklearn (entrada menor).

**Trade-off explícito**: na tabela `model_comparison_full.csv`, a MLP recebe **todas as colunas pós-OHE** enquanto Random Forest/Gradient Boosting recebem apenas as 25 selecionadas. Comparações entre eles devem considerar esse fato — não é "modelo A vs modelo B" estritamente igual.

#### Bundle de inferência — 3 arquivos

Cada run FE com MLP gera, em `PATH_MODEL`, um conjunto com prefixo `pytorch_mlp_<objective>_<run_ts>`:

| Arquivo | O que contém | Para quê serve |
|---------|--------------|----------------|
| `<prefix>.pt` | `state_dict` salvo com `torch.save` | Pesos aprendidos. **Não** contém a classe — só os tensores. |
| `<prefix>_preprocess.joblib` | `ColumnTransformer` **já ajustado** no treino | Aplica a mesma normalização/OHE em produção. |
| `<prefix>_meta.json` | `hidden_dims`, `dropout`, `n_features_in`, `decision_threshold`, `feature_columns_in_order`, `feature_groups`, `best_epoch`, `best_val_loss`, `metrics_test/val`, `torch_version` | Recria a arquitetura **exatamente** + contrato de colunas + threshold de produção. |

**Por que 3 arquivos separados?**
- **Separação de responsabilidades**: pesos (PyTorch) ≠ pré-processamento (sklearn) ≠ contrato (JSON legível).
- O `state_dict` puro é menor e mais portátil que serializar a classe inteira via `pickle`.
- O `_meta.json` é **inspecionável por humanos** — útil para auditoria e debug.

#### Inferência em produção (`predict_with_mlp`)

Quando `POST /predict` chega para um run promovido com `inference_backend="mlp"`:

1. `processor_service._prepare_prediction_features` aplica `ChurnFeatures.build` (mesma strategy do treino), com `monthly_median` **reidratado** de `pipeline_runs.metrics["strategy_monthly_charges_median"]`.
2. `load_mlp_bundle(prefix)`:
   - Lê os 3 arquivos.
   - Reconstrói `_MLPBinary(n_features=meta.n_features_in, hidden_dims=meta.hidden_dims, dropout=meta.dropout)` — arquitetura idêntica à do treino.
   - `model.load_state_dict(torch.load(<prefix>.pt, map_location="cpu"))`.
   - `model.eval()` — desliga Dropout para inferência.
3. `predict_with_mlp(bundle, df_input, threshold=None)`:
   - Alinha colunas pela ordem em `meta.feature_columns_in_order` — qualquer coluna ausente → `ValueError` (HTTP 400).
   - `bundle.preprocess.transform(df)` → matriz `float32` densa.
   - `torch.no_grad()` + `model(tensor)` → logit.
   - `proba = 1 / (1 + exp(-logit))` → P(churn).
   - `label = 1 if proba >= threshold else 0` (threshold = `bundle.decision_threshold` se não passar override).

#### Comparação MLP vs sklearn no `/predict`

| Aspeto | sklearn (joblib) | MLP (PyTorch) |
|--------|------------------|---------------|
| Artefato em disco | 1 arquivo `.joblib` | Bundle de 3 arquivos com mesmo prefixo. |
| Como o run sinaliza | `pipeline_runs.inference_backend = "sklearn"` | `pipeline_runs.inference_backend = "mlp"` + `metrics.mlp_artifact_prefix` |
| Decisão final | `model.predict(X)` (threshold interno ~0.5) | `proba >= decision_threshold` (vem do `_meta.json`) |
| Customização do threshold | Indireta (precisa recodar o serving) | Direta — basta mudar a env e re-treinar |
| Suporte a GPU | Não | Sim (auto-detect; usa CPU se não houver GPU) |

#### Reprodutibilidade da MLP

Pontos onde sementes são fixadas para garantir resultado idêntico em re-execuções:

| Ponto | Mecanismo |
|-------|-----------|
| Pesos iniciais da rede | `torch.manual_seed(random_state)` antes de instanciar `_MLPBinary`. |
| GPU (se houver) | `torch.cuda.manual_seed_all(random_state)`. |
| Ordem de batches no `DataLoader` | `Generator(seed=random_state)` passado para `DataLoader(generator=...)`. |
| Split treino/val | `train_test_split(stratify=y, random_state=42)`. |
| Sementes do sklearn (CV, modelos) | `random_state=42` em todos os pontos. |

`random_state` vem de `settings.random_state` (env `RANDOM_STATE=42` em `.env_example`).

#### Quando a MLP é ignorada / falha silenciosamente

Em `_run_mlp_torch_mvp`:
- `enable_mlp_torch=False` → passo é pulado com log e o run FE continua só com sklearn.
- `import torch` falha → log `"MLP PyTorch indisponível (import)"` + passo pulado (sklearn segue).
- Treino lança exceção → log com `exc_info=True` + `mlp_torch_result = None` + sklearn segue normal.

Em qualquer um desses casos, o run **não pode** ser promovido com `USE_MLP_FOR_PREDICTION=true` (não haverá bundle MLP). O `promote` falha explicitamente exigindo coerência entre `inference_backend` do run e a env.

---

## 7. Regras de Decisão

### Threshold de decisão

| Configuração | Comportamento |
|--------------|---------------|
| `CLASSIFICATION_DECISION_THRESHOLD` (env, default `0.5`) | P(churn) ≥ threshold → `prediction=1`. |
| Form `decision_threshold` na rota FE | Override por run de treino (afeta métricas de teste reportadas). |
| `decision_threshold` no JSON do trigger Airflow | Idem, por DAG run. |
| Threshold interno do CV/tuning | **Não** alterado pelo decision_threshold — sklearn segue ~0.5 internamente. |

> Implementação: `labels_from_probability_threshold(estimator, X, threshold)` em `src/services/pipelines/binary_decision_threshold.py`.

### Eleição de campeão (FE)

Um run FE só fica `active=true` se vencer o comparador intra-grupo (manual vs Airflow, mesmo `inference_backend`):

1. Mesmos `best_model_name` **e** `optimization_metric` que o campeão atual.
2. `cv_<optimization_metric>` (`cv_recall` quando otimização é recall) **estritamente maior** que o campeão atual.

Se as condições não forem atendidas, o run é registrado em `pipeline_runs` mas marcado `active=false` — **não entra** no `/predict` por este fluxo.

### Promoção para produção

`POST /v1/processor/admin/promote` exige **exatamente um** run FE candidato (`active=true`, `status=completed`, `inference_backend = "mlp" if USE_MLP_FOR_PREDICTION else "sklearn"`, `objective = OBJECTIVE`). Em produção (`ENVIRONMENT=prod`) exige também `is_airflow_run=true`.

### Interpretação da saída

| Campo | Significado |
|-------|------|
| `prediction = 1` | Cliente sinalizado como **risco de churn** → recomendado encaminhar para retenção. |
| `prediction = 0` | Cliente classificado como **não-churn** segundo o threshold corrente. |
| `probability` (0–100) | P(churn) percentual. Use como **score** para priorizar listas de retenção. |

> Recomendação operacional: para listas priorizadas, **ordene por `probability` decrescente**, independente do `prediction` binário.

---

## 8. Entradas e Saídas

### Entrada do treino (CSV)

| Regra | Valor |
|-------|-------|
| Última coluna | **target** (binarizado: `>0 → 1`, `0 → 0`) |
| Coluna de ID | **Não enviar** (ex.: remover `customerID`) |
| Header | Linha 1 |
| Tipo de problema | Apenas classificação binária |
| Valores ausentes | Tratados pós-split: mediana (numéricos) / moda (categóricos) |

### Entrada do `/predict` (JSON)

Schema discriminado por `domain` (Pydantic — `src/schemas/processor_schemas.py`).

#### Domínio `churn` (`ChurnFeaturesInput`)
Atributos brutos do CSV de treino, **antes** das features criadas pela strategy (sem `is_new_customer`, `tenure_log`, etc.) e **antes** do `ColumnTransformer` (sem one-hot, sem normalização).

| Campo | Tipo | Restrição |
|-------|------|-----------|
| `gender` | `str` | Ex.: `Male`, `Female` |
| `seniorcitizen` | `int` | 0 ou 1 |
| `partner`, `dependents` | `int` | 0 ou 1 |
| `tenure` | `int` | ≥ 0 |
| `phoneservice` | `int` | 0 ou 1 |
| `multiplelines` | `int \| str` | 0/1 ou texto |
| `internetservice` | `str` | `DSL`, `Fiber optic`, `No` |
| `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies` | `int` | 0 ou 1 |
| `contract` | `str` | `Month-to-month`, `One year`, `Two year` |
| `paperlessbilling` | `int` | 0 ou 1 |
| `paymentmethod` | `str` | Ex.: `Electronic check`, `Credit card (automatic)` |
| `monthlycharges`, `totalcharges` | `float` | ≥ 0 |

> Enviar colunas pós-OHE/escala (ex.: `internetservice_Fiber optic`) → erro 400.

### Saída do `/predict` (JSON)

```json
{
  "id": 42,
  "domain": "churn",
  "pipeline_run_id": 7,
  "prediction": 1,
  "probability": 73.45,
  "input_data": { "...": "..." }
}
```

| Campo | Significado |
|-------|------|
| `id` | PK em `predictions` (auditoria). |
| `pipeline_run_id` | Run que serviu a inferência. |
| `prediction` | 0 ou 1 (após threshold). |
| `probability` | P(classe positiva) × 100, arredondada. |
| `input_data` | Eco do payload original (rastreabilidade / drift). |

### Saída do treino

- **Baseline**: CSV pré-processado (`baseline_sample.csv`, `sample_schema=raw_clean`) + cabeçalhos `X-Pipeline-*`. Joblib em `src/artifacts/models/baseline_model_<obj>_<ts>.joblib`.
- **Feature Engineering**: ZIP `fe_artifacts_<run_id>_<ts>.zip` com:
  - `20_feature_engineering/best_<obj>_<ts>.joblib`
  - `20_feature_engineering/pytorch_mlp_<obj>_<ts>.{pt, _preprocess.joblib, _meta.json}` (se MLP habilitada)
  - `20_feature_engineering/fe_export/model_comparison_full.{csv,md}`
  - `20_feature_engineering/fe_export/{train,test}_features_pre_transform.csv`
  - `20_feature_engineering/fe_export/{train,test}_model_input.csv`
  - `manifest.json` do bundle.

---

## 9. Como Executar o Projeto

### Pré-requisitos
- Docker ≥ 24, Docker Compose ≥ 2.20.
- Portas livres: `8000` (API), `8080` (Airflow), `5432` (Postgres), `5050` (pgAdmin), `8888` (Dozzle).
- Linux/WSL com `id -u` disponível (para `AIRFLOW_UID`).

### Setup do ambiente

```bash
git clone <url-do-repositorio>
cd Machine-Learning

cp .env_example .env
# Preencher: DATABASE_USER, DATABASE_PASS, DATABASE_NAME, SECRET, PGADMIN_*, OBJECTIVE=churn
echo "AIRFLOW_UID=$(id -u)" >> .env

mkdir -p ml_data/uploads
cp /caminho/para/WA_Fn-UseC_-Telco-Customer-Churn.csv ml_data/uploads/
```

Variáveis críticas no `.env`:

| Variável | Padrão | Observação |
|----------|--------|-----------|
| `OBJECTIVE` | `churn` | Domínio servido pelas rotas que não recebem `objective` no form. |
| `USE_MLP_FOR_PREDICTION` | `false` | `true` → próximos runs FE servem PyTorch MLP. |
| `CLASSIFICATION_DECISION_THRESHOLD` | `0.5` | Threshold P(positiva) para `prediction`. |
| `RANDOM_STATE` | `42` | Reprodutibilidade. |
| `SECRET` | _gerar_ | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `ENVIRONMENT` | `development` | `production` desabilita treino síncrono via API. |

### Subir os serviços

```bash
docker compose up --build
```

| Serviço | URL | Descrição |
|---------|-----|-----------|
| API | http://localhost:8000/docs | Swagger interativo |
| Airflow | http://localhost:8080 | UI (user `airflow` / pass `airflow`) |
| pgAdmin | http://localhost:5050 | Interface do PostgreSQL |
| Dozzle | http://localhost:8888 | Logs em tempo real |

Reset completo (apaga DB + volumes nomeados):
```bash
docker compose down -v
```

### Execução passo a passo (caminho rápido — Airflow + auto_promote)

```bash
# 1. (Airflow UI) Em Admin → Variables, importar
#    airflow/bootstrap/ml_training_pipeline_conf.json (ou setar auto_promote=true)

# 2. Disparar via API → DAG (mais simples para o admin):
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/authenticate \
  -F "username=admin@exemplo.com" -F "password=senha" | jq -r .access_token)

curl -X POST http://localhost:8000/v1/processor/admin/train/trigger-dag \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv" \
  -F "optimization_metric=recall" \
  -F "time_limit_minutes=10"

# 3. Acompanhar em http://localhost:8080/dags/ml_training_pipeline/grid

# 4. Promover (se auto_promote=false):
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer $TOKEN"

# 5. Inferência:
curl -X POST http://localhost:8000/v1/processor/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Execução local (sem Docker)

```bash
make install-dev          # pip install -e ".[dev]"
make lint                 # ruff
make test                 # pytest com cobertura
make run                  # uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Estrutura de pastas

```text
Machine-Learning/
├── src/
│   ├── api/v1/endpoints/        # FastAPI (auth, users, roles, processor, health)
│   ├── core/                    # configs, deps, auth, middleware, logging, graphs
│   ├── models/                  # ORM SQLAlchemy
│   ├── schemas/                 # Pydantic
│   ├── services/
│   │   ├── auth/  user/  roles/
│   │   ├── pipelines/
│   │   │   ├── baseline.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── mlp_torch_tabular.py     # MLP PyTorch (treino)
│   │   │   ├── mlp_inference.py         # MLP PyTorch (serving)
│   │   │   ├── fe_model_selection.py
│   │   │   ├── fe_hyperparameter_tuning.py
│   │   │   └── feature_strategies/      # ChurnFeatures, HeartDiseaseFeatures
│   │   └── processor/                   # processor_service, deployment_service, airflow_persistence
│   ├── data/                    # CSVs, pre_processed/, logs/<ts>/
│   ├── artifacts/               # models/ (joblib + bundle MLP), mlruns/, reports/
│   ├── graphs/
│   └── scripts/maintenance/     # drift_report.py, latency_report.py
├── airflow/
│   ├── dags/ml_training_pipeline.py
│   └── bootstrap/               # Variables defaults + script init
├── docker/                      # Dockerfiles + requirements
├── docs/                        # MODEL_CARD.md, OBSERVABILIDADE_E_MANUTENCAO.md, etc.
├── init_db/                     # database.sql (schema + seed)
├── ml_data/uploads/             # CSVs (bind mount)
├── tests/
│   ├── api/  models/  schemas/  smoke/  services/pipelines/
├── reference/TC_01.pdf          # PDF do desafio
├── pyproject.toml               # single source of truth
├── Makefile
├── docker-compose.yaml
├── main.py                      # entrypoint FastAPI
└── .env_example
```

### Monitoramento offline

```bash
# Latência (SLO p95 /predict < 300ms)
python src/scripts/maintenance/latency_report.py --slo-ms 300

# Drift (PSI por feature) — exportar predictions do Postgres antes
python src/scripts/maintenance/drift_report.py \
  --train-csv data/<treino>.csv \
  --predictions-csv exports/predictions.csv
```

Saídas em `src/artifacts/reports/`.

---

## 10. Melhorias Futuras

### Pontos claros de evolução

| Área | Melhoria proposta |
|------|------------------|
| **MLP** | Hyperparameter tuning automático (Optuna / Ray Tune) sobre `hidden_dims`, `dropout`, `lr`, `weight_decay`, `batch_size`. |
| **MLP** | Calibração de probabilidades (`CalibratedClassifierCV` para sklearn / temperature scaling para MLP). |
| **MLP** | Exportar bundle como **TorchScript / ONNX** para reduzir dependência da classe `_MLPBinary` no serving. |
| **Engenharia de features** | Análise de fairness por subgrupo (gênero, senior, dependents) com métricas demográficas. |
| **Engenharia de features** | Detecção formal de drift de schema (ex.: `pandera` no `/predict`) com erro 422 explícito para categorias novas. |
| **Modelagem** | Ensembles (stacking sklearn × MLP), threshold calibrado por custo de FN/FP. |
| **Avaliação** | Validação temporal (out-of-time) em vez de holdout aleatório, quando houver coluna de tempo. |
| **API** | Endpoint `/predict/batch` para listas, com paginação e backpressure. |
| **API** | Validação de schema com `pandera` na entrada do `/predict` (não só Pydantic). |
| **Observabilidade** | Métricas Prometheus + dashboard Grafana (latência, error rate, distribuição de `probability`, contagem por `prediction`). |
| **Observabilidade** | Alertas em tempo real para drift PSI (atualmente offline). |
| **MLflow** | Migrar de SQLite local para servidor MLflow remoto (Postgres + S3) para colaboração. |
| **Testes** | Cobertura ≥ 80% nos serviços; testes de regressão de métricas (golden numbers). |
| **CI/CD** | Pipeline GitHub Actions: ruff + pytest + build da imagem + push para registry. |

### O que seria necessário para produção real

| Categoria | Item |
|-----------|------|
| **Segurança** | Rotação de `SECRET` JWT, `HTTPS` com TLS terminado em load balancer, *rate limiting* por usuário, isolamento de credenciais (Vault / Secrets Manager). |
| **Escalabilidade** | API atrás de load balancer com múltiplos workers `gunicorn`/`uvicorn`; treino exclusivamente em Airflow worker dedicado (Celery/Kubernetes Executor). |
| **Persistência** | Postgres gerenciado (RDS/Cloud SQL) com backup + read replicas; volume de modelos em S3/GCS com versionamento. |
| **Observabilidade** | Tracing distribuído (OpenTelemetry), alertas em PagerDuty/Opsgenie, runbook de incidentes (atualmente em `docs/OBSERVABILIDADE_E_MANUTENCAO.md`). |
| **Compliance / Privacidade** | Anonimização de PII em `predictions.input_data`, retenção configurável, auditoria por LGPD. |
| **MLOps** | Promote automatizado com regras de drift + métrica + smoke test em pre-prod; canário entre `archived` e `active`. |
| **Modelos** | Re-treino agendado (mensal) com gate em queda de PR-AUC; comparação automatizada com modelo em produção em janela móvel. |
| **Deploy em nuvem** | Itens previstos como **bônus opcional do desafio** (AWS / Azure / GCP) — não implementados nesta entrega. |
| **Reprodutibilidade** | Lock de dependências (`pip-tools`/`uv`), pin de imagem base por digest, registry interno. |

---

## Anexos

- **Model Card**: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- **Observabilidade**: [`docs/OBSERVABILIDADE_E_MANUTENCAO.md`](docs/OBSERVABILIDADE_E_MANUTENCAO.md)
- **Documentação consolidada da solução**: [`docs/DOCUMENTACAO_SOLUCAO.md`](docs/DOCUMENTACAO_SOLUCAO.md)
- **Desafio (PDF)**: [`reference/TC_01.pdf`](reference/TC_01.pdf)
