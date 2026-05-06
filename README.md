# Tech Challenge — Previsão de Churn com Rede Neural (MLP / PyTorch)

Easy-run : [https://www.youtube.com/watch?v=_9VqWvDYOuk](https://youtu.be/_9VqWvDYOuk?si=Yzz67Hp_QF9zgQog)

Pipeline End-to-End de classificação binária para **churn** em telecomunicações: ingestão → baseline → feature engineering → MLP PyTorch → API de inferência (FastAPI) → orquestração (Airflow) → tracking (MLflow) → monitoramento (drift / latência).

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Escopo e Limitações](#2-escopo-e-limitações)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Fluxos de Execução](#4-fluxos-de-execução)
5. [Engenharia de Features](#5-engenharia-de-features)
6. [Modelo de Machine Learning](#6-modelo-de-machine-learning)
7. [Regras de Decisão](#7-regras-de-decisão)
8. [Entradas e Saídas](#8-entradas-e-saídas)
9. [Como Executar o Projeto](#9-como-executar-o-projeto)
10. [Melhorias Futuras](#10-melhorias-futuras)

---

## 1. Visão Geral

- **Problema**: operadora de telecom com cancelamento acelerado precisa identificar clientes em risco de churn para acionar retenção.
- **Objetivo**: treinar uma **MLP (PyTorch)**, comparar com baselines (`Logistic Regression` + `Decision Tree` / `Random Forest` / `SVM` / `Gradient Boosting`) sob `StratifiedKFold` 5-fold + holdout estratificado, e servi-la via API autenticada com tracking MLflow.
- **Contexto** (PDF `reference/TC_01.pdf`): Tech Challenge — bibliotecas obrigatórias **PyTorch · scikit-learn · MLflow · FastAPI**; boas práticas: seeds fixos, CV estratificada, Model Card, testes (smoke / schema / API), logging estruturado, **ruff** sem erros, `pyproject.toml` como single source of truth.
- **Dataset**: Telco Customer Churn (IBM) — `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## 2. Escopo e Limitações

### O que o sistema FAZ
- Pipeline Baseline (`LogisticRegression` + `class_weight='balanced'`, pré-processamento pós-split sem leakage).
- Pipeline Feature Engineering com strategy de domínio (`ChurnFeatures`), comparação de 4 modelos sklearn, tuning com `ParameterSampler` + time-budget + guardrails (`min_precision`, `min_roc_auc`).
- Treino paralelo da **MLP PyTorch** (early stopping, `BCEWithLogitsLoss`, `AdamW`).
- Persistência: joblib sklearn + bundle MLP (`.pt` + `_preprocess.joblib` + `_meta.json`) + `manifest.json` + ZIP do run + MLflow.
- API FastAPI com JWT, `/predict`, rotas admin (treino, runs, promote, rollback, history) e `/health`.
- DAG Airflow `ml_training_pipeline` (Baseline → FE → promote opcional).
- Scripts offline de **drift PSI** e **latência** (SLO p95 < 300ms).

### O que o sistema NÃO FAZ
- Não suporta regressão nem multiclasse — apenas binária.
- Não estima LTV nem causalidade.
- Não monitora drift em tempo real (relatório PSI é offline).
- Não aplica reamostragem (SMOTE/undersampling) — desbalanceamento mitigado por `class_weight='balanced'` no baseline e otimização orientada a `recall`.
- Não valida deriva de schema em runtime — categorias novas são absorvidas por `OneHotEncoder(handle_unknown="ignore")`.

### Limitações técnicas
- Cohort estático do Telco (~7.043 linhas, EUA).
- MLP de **arquitetura fixa** (`hidden_dims=(64, 32)`, `dropout=0.0`); sem HPO automatizada.
- `inference_backend` é decidido no momento do **treino** (env `USE_MLP_FOR_PREDICTION`) e gravado no run; alternar a env afeta apenas runs futuros.
- Em `ENVIRONMENT=prod`, treino síncrono via API é **desabilitado** — somente Airflow.
- Caminhos cross-container (Airflow ↔ API) são remapeados em runtime.

### Suposições
- Custo de **FN >> FP** em telecom (5–10×) → métrica de otimização padrão é `recall`.
- CSV de treino chega no contrato: **última coluna = target**, **sem coluna de ID**, classificação binária.

---

## 3. Arquitetura do Sistema

### Componentes

| Componente | Responsabilidade | Localização |
|-----------|-----------------|-------------|
| API FastAPI | JWT, `/predict`, rotas admin, `/health` | `src/api/v1/`, `main.py` |
| Pipeline Baseline | EDA, qualidade, target, split, LR, manifest + sample | `src/services/pipelines/baseline.py` |
| Pipeline FE | Strategy → comparação → tuning → MLP PyTorch → bundle | `src/services/pipelines/feature_engineering.py` |
| MLP PyTorch | Definição + treino + serving | `mlp_torch_tabular.py`, `mlp_inference.py` |
| Feature Strategies | Lógica por domínio (`ChurnFeatures`) | `src/services/pipelines/feature_strategies/` |
| Processor / Deployment | Persistência de runs, eleição de campeão, promote, rollback | `src/services/processor/` |
| PostgreSQL | `users`, `roles`, `pipeline_runs`, `deployed_models`, `predictions` | `init_db/database.sql` |
| MLflow | Tracking SQLite no volume `ml_shared` | `src/artifacts/mlruns/` |
| Airflow | DAG `ml_training_pipeline` (LocalExecutor) | `airflow/dags/` |
| Scripts manutenção | Drift PSI + latência | `src/scripts/maintenance/` |

### Fluxo ponta a ponta

```text
CSV bruto ──> ml_data/uploads/  ──>  /var/www/ml_shared/uploads (API)
                                ──>  /opt/airflow/ml_project/uploads (Airflow)

[Baseline]    csv → split → ColumnTransformer → LogisticRegression
              └─> baseline_sample.csv (raw_clean) + manifest.json + .joblib
              └─> MLflow: <obj>_baseline

[FE]          manifest → strategy.build → split → 4 modelos sklearn (CV 5-fold)
              → seleção (cv_<metric> + guardrails) → tuning ParameterSampler
              → MLP PyTorch (mesmo ColumnTransformer, sem SelectKBest)
              └─> best_<obj>_<ts>.joblib + bundle MLP (.pt + preprocess + meta)
              └─> MLflow: <obj>_feature_engineering

[Promote]     pipeline_runs.active=true → DeployedModels.status=active
              (UNIQUE constraint: 1 ativo por domínio)

[Predict]     POST /predict → resolve deployment ativo →
              inference_backend=="mlp" ? bundle MLP : joblib sklearn
              → predict + predict_proba → grava em predictions
```

---

## 4. Fluxos de Execução

### 4.1 Fluxo Manual (API)

| # | Ação | Endpoint | Notas |
|---|------|----------|-------|
| 1 | Login | `POST /v1/auth/authenticate` | Form (`username`, `password`) → `access_token` |
| 2 | Treino baseline (admin) | `POST /v1/processor/admin/train/baseline` | Form `file=@<csv>` |
| 3 | Treino FE (admin) | `POST /v1/processor/admin/train/feature-engineering` | Form `optimization_metric=recall`, `time_limit_minutes=N`. Lê manifest do baseline ativo. |
| 4 | Listar runs | `GET /v1/processor/admin/runs?pipeline_type=feature_engineering&status=completed` | Filtros opcionais |
| 5 | Promover | `POST /v1/processor/admin/promote` | Promove o único FE `active=true` coerente com `USE_MLP_FOR_PREDICTION` |
| 6 | Predizer | `POST /v1/processor/predict` | JSON com `domain` + `features` (qualquer usuário autenticado) |
| 7 | Rollback (opcional) | `POST /v1/processor/admin/rollback` | Reativa último deployment `archived` |

> Em `ENVIRONMENT=production`, as rotas síncronas de treino retornam erro — usar Airflow.

### 4.2 Fluxo Automatizado (Airflow)

DAG **`ml_training_pipeline`** (LocalExecutor, `schedule_interval=None`):

`validate_input → deactivate_manual_runs → run_baseline → run_fe → promote_fe_optional → notify_complete`

- **Configuração**: Variable `ml_training_pipeline_conf` (JSON) + `dag_run.conf` (override). Bootstrap: `airflow/bootstrap/ml_training_pipeline_conf.json`.
- **Gatilho via API**: `POST /v1/processor/admin/train/trigger-dag` (form com CSV) — grava no volume e dispara o DAG.
- **Gatilho via UI**: `http://localhost:8080/dags/ml_training_pipeline/grid → Trigger DAG`.
- **`auto_promote`**: se `true` no JSON e o run vencer o comparador `cv_<metric>`, promove automaticamente.

---

## 5. Engenharia de Features

Strategy `ChurnFeatures` (`src/services/pipelines/feature_strategies/churn_features.py`).

### Colunas obrigatórias
`tenure`, `monthlycharges`, `totalcharges`, `contract`, `paymentmethod`, `internetservice`.

### Features criadas (15)

| Feature | Definição | Racional |
|---------|-----------|----------|
| `is_new_customer` | `tenure <= 12` | Clientes novos churnam mais. |
| `tenure_log` | `log1p(tenure)` | Estabiliza assimetria. |
| `contract_stability` | Mapeia contrato → 0/1/2 | Lock-in contratual ordinal. |
| `new_customer_in_mounth_contract` | `is_new` & `Month-to-month` | Combinação de risco. |
| `risk_payment_monthly` | `Electronic check` & `Month-to-month` | Padrão histórico de saída. |
| `new_customer_risk_payment_monthly` | Anterior & `is_new` | Tripla intersecção. |
| `fiber_high_cost` | `Fiber optic` & `monthlycharges > median` | Fibra cara — perfil de churn. |
| `fiber_premium_monthly` | `Fiber optic` & `Month-to-month` | Sem lock-in + caro. |
| `fiber_premium_monthly_new_customer` | Anterior & `is_new` | Tripla intersecção. |
| `avg_ticket` | `totalcharges / (tenure + 1)` | Ticket médio (suaviza tenure 0). |
| `charge_ratio` | `monthlycharges / (avg_ticket + ε)` | Detecta upgrade/downgrade. |
| `num_services` | Soma de 6 serviços | Engagement. |
| `low_engagement` | `num_services <= 2` | Baixo engagement → churn. |
| `high_cost_low_engagement` | Caro & baixo engagement | Cliente caro pouco ancorado. |
| `is_auto_payment` | `paymentmethod` contém `automatic` | Pagamento automático reduz fricção. |

### Pré-processamento pós-strategy
- **Numéricas contínuas**: `SimpleImputer(median)` (se houver nulos) + `StandardScaler`.
- **Numéricas binárias**: `passthrough`.
- **Categóricas**: `SimpleImputer(most_frequent)` + `OneHotEncoder(drop="first", handle_unknown="ignore")`.
- **Filtros (só sklearn)**: `VarianceThreshold(0.0)` + `SelectKBest(f_classif, k=25)`.
- **Estado persistido**: `monthly_median` é gravado em `pipeline_runs.metrics` e reidratado na inferência.

### Riscos
| Risco | Estado |
|-------|--------|
| Leakage | Mitigado — pré-processamento aprendido pós-split. |
| Drift de categorias | `OneHotEncoder(handle_unknown="ignore")` evita falha mas pode degradar silenciosamente — exige PSI. |
| Vieses demográficos | Sem análise por subgrupo (dívida documentada no Model Card). |
| Cohort temporal | Snapshot histórico — pode degradar com sazonalidade. |

---

## 6. Modelo de Machine Learning

### Modelo principal
- **MLP PyTorch** (`Linear(N,64) → ReLU → Dropout → Linear(64,32) → ReLU → Dropout → Linear(32,1)`).
- Loss: `BCEWithLogitsLoss`. Optimizer: `AdamW(lr=1e-3, weight_decay=1e-5)`. Early stopping `patience=20` em `val_loss`.
- Bundle servido: `<prefix>.pt` + `<prefix>_preprocess.joblib` + `<prefix>_meta.json`.

### Baselines comparados
`Logistic Regression` (pipeline baseline) · `Decision Tree` · `Random Forest` · `SVM` · `Gradient Boosting`.

### Treino
1. Split 80/20 estratificado, `random_state=42`.
2. CV `StratifiedKFold(n_splits=5)` no treino.
3. Comparação por `cv_<optimization_metric>` + guardrails.
4. Tuning `ParameterSampler` com time-budget.
5. MLP treinada em paralelo no mesmo run (sem `SelectKBest`).
6. MLflow registra params + métricas + dataset CSV + gráficos.

### Métricas
`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc` (baseline), `cv_<metric>`, `overfitting_gap`.

### Limitações
- MLP com arquitetura fixa, sem HPO.
- Sem calibração de probabilidade.
- MLP vê todas as features pós-OHE; sklearn vê só as `k=25` selecionadas — comparações não são estritamente iguais.

> **Documentação técnica detalhada da MLP**: ver [`docs/MLP_PYTORCH.md`](docs/MLP_PYTORCH.md) (arquitetura camada a camada, loop de treino, hiperparâmetros, bundle, inferência, reprodutibilidade).

---

## 7. Regras de Decisão

### Threshold
| Configuração | Comportamento |
|--------------|---------------|
| Env `CLASSIFICATION_DECISION_THRESHOLD` (default `0.5`) | P(churn) ≥ threshold → `prediction=1`. |
| Form `decision_threshold` na rota FE | Override por run. |
| `decision_threshold` no conf do Airflow | Override por DAG run. |
| CV / tuning interno | Não afetado — sklearn usa ~0.5 internamente. |

Implementação: `labels_from_probability_threshold` em `src/services/pipelines/binary_decision_threshold.py`.

### Eleição de campeão (FE)
Run só fica `active=true` se vencer **intra-grupo** (manual vs Airflow, mesmo `inference_backend`):
1. Mesmos `best_model_name` **e** `optimization_metric` que o campeão atual.
2. `cv_<optimization_metric>` estritamente maior.

### Promote
`POST /admin/promote` exige **exatamente um** run FE candidato (`active=true`, `status=completed`, `inference_backend` coerente com `USE_MLP_FOR_PREDICTION`, `objective = OBJECTIVE`). Em `ENVIRONMENT=prod` exige `is_airflow_run=true`. Antigo `active` vira `archived`. Constraint UNIQUE garante 1 ativo por domínio.

### Interpretação
- `prediction=1` → cliente em risco — encaminhar para retenção.
- `probability` (0–100) → **score** para priorização (ordenar listas por probability decrescente).

---

## 8. Entradas e Saídas

### Entrada do treino (CSV)
- Última coluna = **target** (binarizado: `>0 → 1`).
- **Sem coluna de ID** (remover `customerID`).
- Header na linha 1.
- Apenas binária.

### Entrada do `/predict` (JSON)
Schema discriminado por `domain` (`src/schemas/processor_schemas.py`).

**Domínio `churn`** (`ChurnFeaturesInput`): atributos brutos do CSV, **antes** das features criadas pela strategy e **antes** do `ColumnTransformer` (sem one-hot, sem normalização). Campos: `gender`, `seniorcitizen`, `partner`, `dependents`, `tenure`, `phoneservice`, `multiplelines`, `internetservice`, `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`, `contract`, `paperlessbilling`, `paymentmethod`, `monthlycharges`, `totalcharges`.

> Enviar colunas pós-OHE (ex.: `internetservice_Fiber optic`) → erro 400.

### Saída do `/predict`

A resposta tem **4 blocos**: identificação · predição · eco do payload · relatório de auditoria.

```json
{
  "id": 1,
  "domain": "churn",
  "pipeline_run_id": 2,
  "prediction": 0,
  "probability": 5.07,

  "input_data": {
    "gender": "Male", 
    "seniorcitizen": 0, 
    "partner": 0, 
    "dependents": 0,
    "tenure": 72, 
    "phoneservice": 1,
     "multiplelines": 1,
    "internetservice": "Fiber optic",
    "onlinesecurity": 0, 
    "onlinebackup": 1, 
    "deviceprotection": 1,
    "techsupport": 1, 
    "streamingtv": 1, 
    "streamingmovies": 1,
    "contract": "Two year", 
    "paperlessbilling": 0,
    "paymentmethod": "Bank transfer (automatic)",
    "monthlycharges": 109.2, 
    "totalcharges": 7878.3
  },

  "inference_report": {
    "inference_backend": "sklearn",
    "predict_model": "sklearn_pipeline",
    "sklearn_benchmark_classifier": "Gradient Boosting",
    "optimization_metric": "recall",
    "best_cv_score": 0.5258,
    "classification_decision_threshold": 0.3,

    "served_holdout_metrics": {
      "accuracy": 0.7672, 
      "precision": 0.5449, 
      "recall": 0.7460,
      "f1": 0.6298, 
      "roc_auc": 0.8450
    },

    "baseline_holdout_metrics": {
      "accuracy": 0.7395,
      "precision": 0.5060,
      "recall": 0.7888,
      "f1": 0.6165, 
      "roc_auc": null
    },
    "baseline_reference": {
      "role": "baseline_sklearn",
      "baseline_pipeline_run_id": 1,
      "classification_decision_threshold": 0.5,
      "test_accuracy": 0.7395, "test_precision": 0.5060, "test_recall": 0.7888,
      "test_f1": 0.6165, "test_roc_auc": null, "test_pr_auc": 0.6350,
      "description": "Métricas de teste do pipeline Baseline (sklearn). Referência antes do feature engineering; não é o modelo servido em /predict."
    },

    "fe_model_comparison": [
      { "Modelo": "Gradient Boosting",         
      "Origem": "sklearn (pré-tuning)",           
       "Accuracy": 0.7608, 
       "Precision": 0.5358, 
       "Recall": 0.7406, 
       "F1": 0.6218, 
       "ROC AUC": 0.8428 },

      { "Modelo": "Decision Tree",            
       "Origem": "sklearn (pré-tuning)",           
        "Accuracy": 0.7331, 
        "Precision": 0.4975, 
        "Recall": 0.5428, 
        "F1": 0.5192, 
        "ROC AUC": 0.6729 },

      { "Modelo": "Random Forest",            
       "Origem": "sklearn (pré-tuning)",           
        "Accuracy": 0.7473, 
        "Precision": 0.5171, 
        "Recall": 0.7273, 
        "F1": 0.6044, 
        "ROC AUC": 0.8208 },

      { "Modelo": "SVM",                       
      "Origem": "sklearn (pré-tuning)",           
       "Accuracy": 0.7970, 
       "Precision": 0.6341, 
       "Recall": 0.5561,
        "F1": 0.5926, 
        "ROC AUC": 0.7876 },

      { "Modelo": "Gradient Boosting (tuned)",
       "Origem": "sklearn (pós-tuning, promovido)",
        "Accuracy": 0.7672, 
        "Precision": 0.5449, 
        "Recall": 0.7460, 
        "F1": 0.6298, 
        "ROC AUC": 0.8450 },

      { "Modelo": "PyTorch MLP",              
       "Origem": "rede neural (teste)",       
             "Accuracy": 0.7970,
              "Precision": 0.6333,
               "Recall": 0.5588, 
               "F1": 0.5938, 
               "ROC AUC": 0.8454 }
    ],

    "mlp_training_summary": {
      "hidden_dims": [64, 32], 
      "dropout": 0,
      "lr": 0.001,
       "weight_decay": 1e-5,
      "batch_size": 64,
       "max_epochs": 300, 
       "early_stopping_patience": 20,
      "val_fraction": 0.15,
      "best_epoch": 14, 
      "best_val_loss": 0.4169
    },

    "summary_lines": [
      "Backend: sklearn · modelo declarado: sklearn_pipeline.",
      "Classificador de referência (CV / estudo no FE): Gradient Boosting.",
      "Melhor score de CV (recall): 0.5258.",
      "Teste holdout — Baseline vs servido (FE sklearn) — Recall: 0.7888 → 0.7460."
    ]
  }
}
```

#### Como ler esta resposta

**Bloco 1 — Identificação**

| Campo | Significado |
|-------|-------------|
| `id` | PK em `predictions` (rastreabilidade da chamada). |
| `domain` | Domínio do modelo (`churn`). |
| `pipeline_run_id` | Run que serviu a inferência — ligar com `/admin/runs`. |

**Bloco 2 — Predição**

| Campo | Significado |
|-------|-------------|
| `prediction` | `0` ou `1` após aplicar `classification_decision_threshold`. |
| `probability` | P(churn) **em percentual** (0–100). No exemplo: `5.07` = 5,07% de risco. |

> Exemplo: cliente com `tenure=72` (6 anos), contrato de 2 anos, pagamento automático e múltiplos serviços ativos → perfil "fiel" → P(churn) ≈ 5%. `prediction=0` é coerente.

**Bloco 3 — `input_data`**

Eco do payload original (auditoria + drift). Útil para reconstruir o caso e correlacionar com PSI.

**Bloco 4 — `inference_report`** (campos de auditoria)

| Subcampo | O que mostra |
|----------|--------------|
| `inference_backend` | `"sklearn"` (joblib) ou `"mlp"` (bundle PyTorch) — qual artefato gerou a predição. |
| `predict_model` | `sklearn_pipeline` ou `pytorch_mlp` — etiqueta do modelo servido. |
| `sklearn_benchmark_classifier` | Algoritmo sklearn vencedor da comparação (no exemplo: Gradient Boosting tunado). |
| `optimization_metric` | Métrica que dirigiu seleção e tuning no FE (no exemplo: `recall`). |
| `best_cv_score` | Score em CV 5-fold da métrica de otimização (`cv_recall = 0.5258`). |
| `classification_decision_threshold` | Threshold P(positiva) usado **neste run** (no exemplo: `0.3` — agressivo para recall). |
| `served_holdout_metrics` | Métricas no **holdout do treino** (test_size=20%) do modelo **promovido**. |
| `baseline_holdout_metrics` / `baseline_reference` | Métricas do **Baseline anterior** (Logistic Regression). Permite calcular ganho do FE. |
| `fe_model_comparison` | Tabela com **todos** os modelos avaliados no run FE: 4 sklearn pré-tuning + sklearn pós-tuning (promovido) + MLP PyTorch. |
| `mlp_training_summary` | Hiperparâmetros + `best_epoch` / `best_val_loss` da MLP — só presente quando `enable_mlp_torch=True`. |
| `summary_lines` | Resumo legível, pronto para colar em ticket / Slack. |

> Os valores em `served_holdout_metrics`, `baseline_holdout_metrics` e `fe_model_comparison` referem-se ao **conjunto de teste do treino** daquele run, **não** ao indivíduo do pedido. Servem como prova de qualidade do modelo, não de confiança da predição individual (essa é dada por `probability`).

#### Lendo o exemplo

- **Backend servido**: sklearn (joblib do `Gradient Boosting` tunado).
- **Threshold**: `0.3` (otimizado para recall — captura mais positivos ao custo de mais FP).
- **Ganho do FE sobre o Baseline**: `f1 0.6165 → 0.6298` (+1,3pp); `roc_auc null → 0.8450`. Recall caiu (0.7888 → 0.7460) porque o FE prioriza `f1`/`roc_auc` enquanto otimiza recall em CV.
- **MLP** ficou com melhor `roc_auc` (0.8454 vs 0.8450) e melhor `accuracy/precision`, mas pior `recall` — não foi promovido neste run porque `USE_MLP_FOR_PREDICTION=false` no momento do treino.

### Saída do treino
- **Baseline**: CSV `baseline_sample.csv` (raw_clean) + cabeçalhos `X-Pipeline-*` + joblib em `src/artifacts/models/`.
- **FE**: ZIP `fe_artifacts_<run_id>_<ts>.zip` com joblib campeão, bundle MLP, `model_comparison_full.{csv,md}`, CSVs pré/pós-transform, `manifest.json`.

---

## 9. Como Executar o Projeto

### 9.1 Pré-requisitos

| Item | Versão | Observação |
|------|--------|-----------|
| Docker | ≥ 24 | [docs](https://docs.docker.com/get-docker/) |
| Docker Compose | ≥ 2.20 | Plugin do Docker |
| Linux/WSL ou macOS | — | `id -u` necessário para `AIRFLOW_UID` |
| Portas livres | `8000` (API), `8080` (Airflow), `5432` (Postgres), `5050` (pgAdmin), `8888` (Dozzle) | — |
| Python (opcional) | ≥ 3.10 | Apenas para rodar localmente sem Docker |

### 9.2 Setup do `.env`

```bash
git clone <url-do-repositorio>
cd Machine-Learning

cp .env_example .env
echo "AIRFLOW_UID=$(id -u)" >> .env
```

Editar `.env` e preencher **obrigatoriamente**:

| Variável | Exemplo | Descrição |
|----------|---------|-----------|
| `DATABASE_USER` | `processing` | Usuário do PostgreSQL |
| `DATABASE_PASS` | `<senha>` | Senha do PostgreSQL |
| `DATABASE_NAME` | `processing` | Nome do banco |
| `SECRET` | _gerar_ | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `PGADMIN_EMAIL` | `admin@exemplo.com` | Login do pgAdmin |
| `PGADMIN_PASSWORD` | `<senha>` | Senha do pgAdmin |
| `OBJECTIVE` | `churn` | Domínio servido nas rotas sem `objective` no form |
| `USE_MLP_FOR_PREDICTION` | `true` ou `false` | `true` → próximos runs FE servem PyTorch MLP |
| `CLASSIFICATION_DECISION_THRESHOLD` | `0.5` | Threshold P(positiva) para `prediction` |
| `ENVIRONMENT` | `development` | `production` desativa treino síncrono via API |

> `DATABASE_SERVER=db_processing` já está configurado para uso com Docker — não alterar.

### 9.3 Posicionar o CSV de treino

```bash
mkdir -p ml_data/uploads
cp /caminho/para/WA_Fn-UseC_-Telco-Customer-Churn.csv ml_data/uploads/
```

Esta pasta é **bind-mountada** simultaneamente em:
- `/var/www/ml_shared/uploads` (API)
- `/opt/airflow/ml_project/uploads` (Airflow)

### 9.4 Subir o stack (Docker)

```bash
docker compose up --build
# ou em background:
docker compose up --build -d
```

Ou via Makefile:
```bash
make docker-up      # docker compose up --build
make docker-down    # docker compose down
```

Serviços disponíveis após boot:

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| API (Swagger) | http://localhost:8000/docs | JWT por usuário |
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| pgAdmin | http://localhost:5050 | `PGADMIN_EMAIL` / `PGADMIN_PASSWORD` |
| Dozzle (logs) | http://localhost:8888 | — |
| PostgreSQL | localhost:5432 | `DATABASE_USER` / `DATABASE_PASS` |

Reset completo (apaga DB + volumes nomeados):
```bash
docker compose down -v
```
> Isto não apaga `ml_data/uploads` (bind mount no host).

### 9.5 Verificações pós-boot

```bash
# 1. API alive + DB up
curl -s http://localhost:8000/v1/health | jq
# → {"alive": true, "database": "up", "service": "...", "environment": "development"}

# 2. Airflow operacional
curl -s http://localhost:8080/health | jq
# → {"metadatabase":{"status":"healthy"},"scheduler":{...}}

# 3. DAG carregada
docker exec airflow_scheduler airflow dags list | grep ml_training_pipeline

# 4. CSV visível dentro do Airflow
docker exec airflow_scheduler ls /opt/airflow/ml_project/uploads/

# 5. Tabelas seedadas no Postgres
docker exec database_processing psql -U "$DATABASE_USER" -d "$DATABASE_NAME" \
  -c "SELECT id, name, email, role_id, active FROM users;"
```

### 9.6 Usuários semeados (`init_db/database.sql`)

A inicialização do banco cria **dois usuários administradores** (`role_id=2`):

| `id` | `name` | `email` | Role |
|------|--------|---------|------|
| 1 | Gabriel Drumond | `gabriel.drumond@cod3bit.com.br` | Administrator |
| 2 | airflow | `airflow@airflow.com.br` | Administrator |

> As senhas estão em **bcrypt** no SQL (não em texto plano no repositório). Use uma das opções abaixo conforme o seu cenário:

**Opção A — criar seu próprio admin (recomendado em ambiente novo)**:

```bash
# 1. Signup (cria com role_id=1 = User)
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Seu Nome","email":"voce@exemplo.com","password":"sua_senha"}'

# 2. Promover para Administrator via SQL
docker exec database_processing psql -U "$DATABASE_USER" -d "$DATABASE_NAME" \
  -c "UPDATE users SET role_id = 2 WHERE email = 'voce@exemplo.com';"
```

**Opção B — gerar novo hash bcrypt e atualizar a seed antes do primeiro `up`**:

```bash
python -c "from passlib.hash import bcrypt; print(bcrypt.hash('nova_senha'))"
# Substituir o hash em init_db/database.sql ANTES do primeiro `docker compose up`
```

> Roles disponíveis: `1=User` (rotas básicas) e `2=Administrator` (rotas `/admin/*` e `/processor/*` de treino/promote).

### 9.7 Caminho rápido — primeiro run + predição

```bash
# 1. Login (capturar token)
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/authenticate \
  -F "username=voce@exemplo.com" -F "password=sua_senha" | jq -r .access_token)

# 2. Disparar DAG via API (Airflow executa Baseline + FE)
curl -X POST http://localhost:8000/v1/processor/admin/train/trigger-dag \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./ml_data/uploads/WA_Fn-UseC_-Telco-Customer-Churn.csv" \
  -F "optimization_metric=recall" \
  -F "time_limit_minutes=10"

# 3. Acompanhar em http://localhost:8080/dags/ml_training_pipeline/grid

# 4. Promover (após o DAG concluir; pular se auto_promote=true no JSON)
curl -X POST http://localhost:8000/v1/processor/admin/promote \
  -H "Authorization: Bearer $TOKEN"

# 5. Inferência
curl -X POST http://localhost:8000/v1/processor/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"domain":"churn","features":{
    "gender":"Female","seniorcitizen":0,"partner":1,"dependents":0,
    "tenure":12,"phoneservice":1,"multiplelines":0,
    "internetservice":"Fiber optic","onlinesecurity":0,"onlinebackup":0,
    "deviceprotection":0,"techsupport":0,"streamingtv":1,"streamingmovies":1,
    "contract":"Month-to-month","paperlessbilling":1,
    "paymentmethod":"Electronic check","monthlycharges":95.5,"totalcharges":1146.0
  }}'
```

### 9.8 Execução local (sem Docker, para dev)

`pyproject.toml` é a single source of truth de dependências. Comandos via Makefile:

```bash
make install-dev    # pip install -e ".[dev]"  (inclui pytest, ruff, mypy)
make lint           # ruff .
make format         # black .
make test           # pytest com cobertura
make coverage       # pytest --cov com relatório HTML
make run            # uvicorn main:app --reload --host 0.0.0.0 --port 8000
make clean          # remove build/dist/cache
```

Pré-condições para rodar local:
- Postgres disponível (use `docker compose up -d db_processing` se quiser apenas o banco).
- `.env` com `DATABASE_SERVER=localhost` (em vez de `db_processing`).
- `PATH_DATA`, `PATH_MODEL`, etc. apontando para diretórios locais (defaults do `.env_example` já funcionam: `src/data/`, `src/artifacts/models/`).

### 9.9 Monitoramento offline

```bash
# Latência (SLO p95 /predict < 300ms)
python src/scripts/maintenance/latency_report.py --slo-ms 300

# Drift PSI (após exportar predictions do Postgres como CSV)
python src/scripts/maintenance/drift_report.py \
  --train-csv data/<treino>.csv \
  --predictions-csv exports/predictions.csv
```

Saídas em `src/artifacts/reports/` (configurável via `PATH_MAINTENANCE_REPORTS`).

### 9.10 Estrutura de pastas

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
│   │   │   ├── mlp_torch_tabular.py
│   │   │   ├── mlp_inference.py
│   │   │   ├── fe_model_selection.py
│   │   │   ├── fe_hyperparameter_tuning.py
│   │   │   └── feature_strategies/
│   │   └── processor/
│   ├── data/                    # CSVs, pre_processed/, logs/<ts>/
│   ├── artifacts/               # models/, mlruns/, reports/
│   ├── graphs/
│   └── scripts/maintenance/
├── airflow/                     # dags/, bootstrap/, plugins/, logs/
├── docker/                      # Dockerfiles + requirements
├── docs/                        # MODEL_CARD.md, MLP_PYTORCH.md, etc.
├── init_db/database.sql         # Schema + seed (users, roles)
├── ml_data/uploads/             # CSVs (bind mount)
├── tests/                       # src/api/, src/models/, src/schemas/, smoke/, src/services/
├── reference/TC_01.pdf
├── pyproject.toml               # single source of truth
├── Makefile                     # install-dev, lint, test, run, docker-*
├── docker-compose.yaml
├── main.py
└── .env_example
```

---

## 10. Melhorias Futuras

### Evolução técnica
- HPO automatizada da MLP (Optuna / Ray Tune) sobre `hidden_dims`, `dropout`, `lr`, `batch_size`.
- Calibração de probabilidades (`CalibratedClassifierCV` / temperature scaling).
- Exportar bundle MLP como **TorchScript / ONNX** (reduz acoplamento com a classe `_MLPBinary`).
- Endpoint `/predict/batch` com paginação.
- Validação de schema com `pandera` no `/predict` (não só Pydantic).
- Detecção formal de drift de schema com erro 422 explícito.
- Análise de fairness por subgrupo (gênero, senior, dependents).
- Dashboard Prometheus + Grafana (latência, error rate, distribuição de `probability`).
- Alertas de drift PSI em tempo real.
- MLflow remoto (Postgres + S3) para colaboração.

### Para produção real
- TLS terminado em load balancer + rate limiting + secrets manager (Vault).
- Postgres gerenciado (RDS/Cloud SQL) + storage de modelos em S3/GCS.
- Treino em Airflow worker dedicado (Celery/Kubernetes Executor).
- Tracing distribuído (OpenTelemetry) + alertas em PagerDuty.
- Anonimização de PII em `predictions.input_data` (LGPD).
- Promote automatizado com gate de drift + métrica + smoke test em pre-prod.
- Re-treino agendado (mensal) com gate em queda de PR-AUC.
- Deploy em nuvem (AWS / Azure / GCP) — bônus opcional do desafio.
- Lock de dependências (`pip-tools` / `uv`) + pin de imagem base por digest.

---

## Anexos

- [`docs/MLP_PYTORCH.md`](docs/MLP_PYTORCH.md) — implementação detalhada da rede neural.
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — Model Card completa (limitações, vieses, cenários de falha).
- [`docs/OBSERVABILIDADE_E_MANUTENCAO.md`](docs/OBSERVABILIDADE_E_MANUTENCAO.md) — plano de monitoramento.
- [`docs/DOCUMENTACAO_SOLUCAO.md`](docs/DOCUMENTACAO_SOLUCAO.md) — documentação consolidada da solução.
- [`reference/TC_01.pdf`](reference/TC_01.pdf) — descrição original do desafio.
