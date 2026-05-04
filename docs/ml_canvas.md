# ML Canvas — Previsão de Churn (Telco IBM)

**Projeto:** Machine-Learning · Tech Challenge Fase 01
**Data:** 2026-05-04
**Domínio:** Classificação binária supervisionada (churn em telecomunicações)
**Dataset:** Telco Customer Churn (IBM) — 7.043 clientes × 21 colunas

---

## Bloco 1 · Proposta de valor

Reduzir a taxa de churn da operadora oferecendo **retenção proativa** aos clientes
com maior risco de cancelamento previstos para os próximos 30 dias. O modelo
classifica cada cliente em *alto* / *baixo* risco e fornece a probabilidade calibrada
para o time de retenção priorizar contatos.

**Para quem:** diretoria comercial e time de retenção (CRM).
**Por quê:** custo de aquisição de novo cliente ≫ custo de retenção; cada
cancelamento evitado preserva LTV inteiro.

## Bloco 2 · Decisões e ações

A predição alimenta um *fluxo de retenção*:

1. Modelo gera score diário para todos os clientes ativos.
2. Clientes com `prob_churn ≥ threshold` (otimizado para recall) entram numa
   fila priorizada por LTV estimado.
3. Time de retenção contata o cliente em até 48h com **oferta personalizada**:
   desconto temporário, upgrade de plano, suporte dedicado.
4. Resultado do contato (aceitou/recusou/sem resposta) volta ao sistema como
   feedback para retreinos.

## Bloco 3 · Aprendizado

- **Tipo de modelo:** classificador binário supervisionado.
- **Treino:** offline em batch via [Airflow DAG](../airflow/dags/ml_training_pipeline.py)
  com retreinos mensais.
- **Comparação:** sklearn (LR, DT, RF, SVM, GB) + MLP PyTorch — vencedor
  promovido para `/predict`.
- **Tracking:** experimentos, params, métricas e dataset version no MLflow.

## Bloco 4 · Predições

**Saída do modelo:**

- `prob_churn ∈ [0, 1]` — probabilidade calibrada do cliente cancelar nos
  próximos 30 dias.
- `decision ∈ {0, 1}` — classificação binária com threshold otimizado.

**Latência:** SLO p95 `/predict` < 300ms (medido por
[middleware](../src/core/middleware/request_record.py)).

**Modo de servimento:** real-time via [API FastAPI](../src/api/v1/endpoints/processor.py)
+ batch noturno para CRM.

## Bloco 5 · Features

**19 features brutas** do dataset Telco IBM:

- *Demográficas:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- *Contratuais:* `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
- *Serviços:* `PhoneService`, `MultipleLines`, `InternetService`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`.
- *Faturamento:* `MonthlyCharges`, `TotalCharges`.

**15 features derivadas** pelo
[`ChurnFeatures` strategy](../src/services/pipelines/feature_strategies/churn_features.py):
`contract_stability`, `is_new_customer`, `tenure_log`, `avg_ticket`,
`charge_ratio`, `risk_payment_monthly`, e outras (ver código).

## Bloco 6 · Coleta de dados

- **Treino atual:** [`telco_churn.csv`](../src/data/telco_churn.csv) — snapshot
  histórico (Telco IBM).
- **Em produção:** extração mensal incremental do CRM com mesmas colunas,
  agregadas por `customer_id` no fechamento mensal.
- **Volume esperado:** ~50 mil clientes ativos / mês após escala.
- **Drift monitorado:** PSI por feature via
  [`drift_report.py`](../src/scripts/maintenance/drift_report.py) (alertas
  warning ≥ 0.10, critical ≥ 0.25).

## Bloco 7 · Construção do modelo

**Pipeline reprodutível:**

1. [`Baseline`](../src/services/pipelines/baseline.py) — `DummyClassifier(strategy='most_frequent')`
   (referência mínima) + `LogisticRegression(class_weight='balanced')` com
   `ColumnTransformer` para imputação e encoding pós-split.
2. [`FeatureEngineering`](../src/services/pipelines/feature_engineering.py) — aplica
   `ChurnFeatures`, treina e tunia DT/RF/SVM-RBF/GB com `StratifiedKFold(5)` +
   `ParameterSampler`.
3. [`mlp_torch_tabular.py`](../src/services/pipelines/mlp_torch_tabular.py) — MLP
   PyTorch (Linear → ReLU → Dropout → Linear, `BCEWithLogitsLoss`, AdamW,
   early stopping pela val loss).

Vencedor por `cv_recall` é candidato ao promote (`POST /v1/processor/admin/promote`).

## Bloco 8 · Avaliação offline

**Métrica primária:** **Recall** (sensibilidade) sobre a classe positiva ≥ 0.70.

Justificativa: em retenção de churn o custo de **falso negativo** (cliente
cancela e não foi contatado) é maior que o de **falso positivo** (oferta de
retenção desnecessária). Recall mede exatamente "fração dos churns capturados".

**Métricas secundárias:**

| Métrica | Alvo |
|---|---|
| ROC-AUC | ≥ 0.80 |
| PR-AUC (Average Precision) | ≥ 0.55 |
| F1-score | ≥ 0.55 |
| Precision | ≥ 0.45 |
| Accuracy | informativa, não decisória |

**Validação:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` +
holdout estratificado 80/20.

**Comparação obrigatória:** todos os modelos contra
`DummyClassifier(strategy='most_frequent')` para garantir aprendizado real
(em desbalanceado ~26.5%, dummy fica preso em recall=0).

## Bloco 9 · Avaliação online

### Métrica de negócio

**Custo de churn evitado por mês (R$):**

```
Custo evitado (R$/mês) = (TP × CLV_mensal_médio × taxa_aceitação)
                       − (FP × custo_oferta_retenção)
                       − (FN × CLV_mensal_médio × perda_evitável)
```

**Componentes:**

| Símbolo | Descrição | Valor referência |
|---|---|---|
| `TP` | Verdadeiros positivos no mês | medido |
| `FP` | Falsos positivos no mês | medido |
| `FN` | Falsos negativos no mês | medido |
| `CLV_mensal_médio` | Receita média preservada por retenção | ~R$ 65 × 32 meses ≈ **R$ 2.080** |
| `custo_oferta_retenção` | Desconto/contato por cliente | **R$ 50** |
| `taxa_aceitação` | Fração que aceita a oferta | **0.30** |
| `perda_evitável` | Fração de FN que seria evitável se contatado | **0.50** |

`CLV_mensal_médio` deriva do dataset (MonthlyCharges médio × tenure médio dos
retidos — calculado no [notebook EDA](../notebooks/01_eda_churn.ipynb)).
Os parâmetros assumidos (R$ 50, 30%, 50%) são **premissas** a calibrar com
dados reais de campanha após o primeiro ciclo em produção. Análise quantitativa
de trade-off FP×FN ficará na Etapa 2 deste Tech Challenge.

### SLO técnico

- **Latência p95** `/predict` **< 300ms** (medido em produção via
  [middleware](../src/core/middleware/request_record.py) e auditado por
  [`latency_report.py`](../src/scripts/maintenance/latency_report.py)).
- **Disponibilidade:** 99.5% mensal.
- **Drift PSI:** alerta em ≥ 0.10 (warning), retreino forçado em ≥ 0.25 (critical).

### Stakeholders

| Papel | Interesse | Sucesso = |
|---|---|---|
| Diretoria comercial | Reduzir taxa de churn mensal | Churn rate −15% YoY |
| Time de retenção | Receber lista priorizada acionável | ≥ 70% recall, fila < 200 contatos/dia |
| Engenharia de ML | Modelo estável em produção | SLO p95 < 300ms, drift < 0.10 |
| Data science | Aprendizado real vs baseline | ROC-AUC ≥ 0.80, recall ≥ 0.70 |

---

## Riscos e limitações

1. **Snapshot estático:** Telco IBM é dataset fechado, não reflete sazonalidade
   real. Em produção, esperar deriva de comportamento (PSI > 0.10) em 3–6 meses.
2. **Premissas de custo não validadas:** parâmetros da fórmula de custo são
   assumidos. Calibrar com primeiro ciclo de retenção real.
3. **Causalidade:** modelo prediz risco, **não** efeito da intervenção. Para
   medir uplift da campanha de retenção, será necessário A/B test (fora do
   escopo da Fase 1).
4. **Threshold único:** atualmente um threshold global. Segmentos (Contract
   M2M vs 2y) podem se beneficiar de thresholds calibrados — backlog.
5. **Recall vs precision:** otimizar para recall implica aceitar FP altos.
   A fórmula de custo penaliza FPs, então o equilíbrio é dinâmico — revisar
   threshold a cada retreino.

## Referências

- [`README.md`](../README.md) — setup e arquitetura do projeto.
- [`docs/RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md) — visão técnica detalhada.
- [`docs/OBSERVABILIDADE_E_MANUTENCAO.md`](OBSERVABILIDADE_E_MANUTENCAO.md) —
  monitoramento e SLOs.
- [`notebooks/01_eda_churn.ipynb`](../notebooks/01_eda_churn.ipynb) — EDA
  exploratória que sustenta este canvas.
