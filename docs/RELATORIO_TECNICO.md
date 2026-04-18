# Relatório Técnico — ML Engineering

## 1. Visão geral do projeto

Plataforma de Machine Learning Engineering para **classificação binária tabulada**, cobrindo o ciclo completo:

```
Dados brutos → Baseline → Feature Engineering → Registro → Promoção → API de Predição → Monitoramento
```

Domínios suportados: `heart_disease` · `churn` (em implementação)

---

## 2. Arquitetura e divisão de responsabilidades

| Camada | Módulo | Responsabilidade |
|--------|--------|-----------------|
| **API** | `api/v1/endpoints/` | Contratos HTTP, autenticação, roteamento |
| **Schemas** | `schemas/` | Validação de entrada e saída (Pydantic) |
| **Serviços de negócio** | `services/processor/` | Orquestração de treino, promoção, predição |
| **Pipelines ML** | `services/pipelines/` | Baseline, Feature Engineering, estratégias por domínio |
| **Persistência** | `models/` | ORM SQLAlchemy (runs, deployments, predições) |
| **Configuração** | `core/` | Settings, auth, logging, middleware |
| **Manutenção** | `scripts/maintenance/` | Relatórios offline de latência e drift |

### Fluxo completo

```
[Admin]
  Upload CSV → POST /train/baseline
                   ↓
            pipeline_runs (baseline, status=completed)
                   ↓
  Upload CSV pré-processado → POST /train/feature-engineering
                   ↓
            pipeline_runs (fe, status=completed)
                   ↓
  POST /admin/promote  →  deployed_models (status=active)
                   ↓
[Consumidor]
  POST /predict  →  predictions (input_data, prediction, probability)
                   ↓
[Operação — offline]
  latency_report.py  →  artifacts/reports/latency_summary_*.csv
  drift_report.py    →  artifacts/reports/drift_psi_*.csv
                   ↓
[Admin — decisão]
  GET /deployments/{domain}/history  →  comparar métricas
  POST /admin/rollback               →  reverter se necessário
```

---

## 3. EDA e preparação de dados (Baseline)

O pipeline `Baseline` executa automaticamente os seguintes passos de EDA e preparação:

### Passo 1 — Carregamento
- Detecta o alvo como **última coluna** do CSV (contrato definido).
- Registra shape, colunas e primeiras linhas nos logs.

### Passo 2 — Visão geral
- `data.info()`, `data.describe()` — estatísticas iniciais nos logs.

### Passo 3 — Missing values
- Contagem e percentual por coluna.
- Gera gráfico `missing_values_<timestamp>.png` se houver ausentes.
- Imputa: **mediana** para numéricos, **moda** para categóricos.

### Passo 4 — Análise do target
- Binariza o alvo: qualquer valor > 0 → 1, 0 → 0.
- Calcula ratio de balanceamento.
- Gera gráficos de distribuição: `target_distribution_pie_*.png` e `target_distribution_bar_*.png`.
- **Alerta** de desbalanceamento se ratio < 0.5 (mitigação: `class_weight='balanced'`).

### Passo 5 — Outliers
- Boxplots por coluna numérica: `outliers_boxplot_<timestamp>.png`.

### Passo 6 — Limpeza e encoding
- Remove colunas de metadado (`dataset`).
- One-hot encoding em colunas categóricas multi-valor.
- Colunas binárias (`True/False`) → `bool`.

---

## 4. Construção e avaliação do modelo

### 4.1 Baseline
- Algoritmo: **Regressão Logística** com `class_weight='balanced'` e seed fixo.
- Métricas calculadas no conjunto de teste (split estratificado 80/20):

| Métrica | Descrição |
|---------|-----------|
| `test_accuracy` | Acurácia geral |
| `test_f1` | F1-Score (média harmônica precision/recall) |
| `test_precision` | Precisão — qualidade dos positivos preditos |
| `test_recall` | Recall — cobertura dos positivos reais |
| `overfitting_gap` | `train_accuracy - test_accuracy` |

- Todos os parâmetros e métricas registrados no **MLflow** (experimento `{objective}_baseline`).

### 4.2 Feature Engineering
- Estratégias de features por domínio (`FeatureStrategy` + `STRATEGY_REGISTRY`).
- Seleção: `SelectKBest` com `f_classif` — top 25 features estatisticamente relevantes.
- Candidatos comparados: Decision Tree · Random Forest · SVM (RBF) · Gradient Boosting.
- Tuning por **orçamento de tempo** (`time_limit_minutes`, padrão 2min) com `ParameterSampler`.
- Importância de features: Gini + Permutation Importance.
- Métricas e gráficos registrados no MLflow (experimento `{objective}_feature_engineering`).

### 4.3 Tabela de comparação (a preencher após execução)

| Modelo | Accuracy | F1 | Precision | Recall | ROC AUC |
|--------|----------|----|-----------|--------|---------|
| Baseline (Logistic Regression) | — | — | — | — | — |
| FE — melhor modelo (a definir) | — | — | — | — | — |

> Preencher com os valores reais após execução nos datasets de heart disease e churn.

---

## 5. Gerenciamento de modelos

| Conceito | Implementação |
|----------|--------------|
| **Registry de runs** | Tabela `pipeline_runs` — histórico de todos os treinos |
| **Modelo ativo** | Tabela `deployed_models` com `status=active` — um por domínio |
| **Histórico de versões** | `GET /admin/deployments/{domain}/history` |
| **Promoção** | `POST /admin/promote` — arquiva o ativo anterior |
| **Rollback** | `POST /admin/rollback` — reativa o archived mais recente |
| **Métricas no deploy** | `metrics_snapshot` — captura métricas do run no momento da promoção |

### Critério de promoção

Promover apenas quando:
1. Métrica principal do candidato supera o ativo em ≥ 2%.
2. PSI médio das features < 0.10 (distribuição estável).
3. Run com `status=completed` e arquivo de modelo existente.

---

## 6. Monitoramento

### Latência
- Registro automático por requisição: `duration_ms` em `logs/api_requests/access.jsonl`.
- Relatório: `python scripts/maintenance/latency_report.py --slo-ms 300`
- **SLO**: p95 de `/predict` < 300ms.

### Drift de dados
- Comparação PSI: distribuição de treino vs predições em produção.
- Relatório: `python scripts/maintenance/drift_report.py --train-csv <ref> --predictions-csv <prod>`
- Thresholds: ok < 0.10 · warning 0.10–0.25 · critical > 0.25.
- **Periodicidade**: após cada promoção e semanalmente.

### Logs em tempo real
- **Dozzle** disponível em `http://localhost:8888` ao subir o compose.
- Logger `api.request` → `access.jsonl` + stdout.
- Logger `ml.pipeline` → arquivo por run + stdout.

---

## 7. Limitações e trabalho futuro

| Limitação | Decisão atual | Evolução futura |
|-----------|--------------|-----------------|
| Classificação binária apenas | Escopo definido para o projeto | Parâmetro `problem_type` + estratégias de regressão |
| Desbalanceamento | `class_weight='balanced'` | SMOTE, undersampling |
| Treino na thread HTTP | Simples para MVP | Workers async (Celery) ou Airflow |
| Drift apenas offline | Scripts manuais/agendados | Alertas automáticos no predict |
| MLflow local (SQLite) | Suficiente para desenvolvimento | Servidor MLflow compartilhado |
| Feature store informal | CSV pré-processado | Feature store formal (Feast, Hopsworks) |
