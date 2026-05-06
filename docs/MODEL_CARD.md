# Model Card — Previsão de Churn (Telco)

> Última atualização: 05/05/2026

---

## 1. Identificação

| Campo | Valor |
|-------|-------|
| Nome do modelo | `churn_classifier` |
| Domínio (objective) | `churn` |
| Versão (`pipeline_run_id`) | _preencher após promote (ex.: 12)_ |
| Backend de inferência | `mlp` (PyTorch) — fallback `sklearn` se `USE_MLP_FOR_PREDICTION=false` |
| Modelos comparados | DummyClassifier, Logistic Regression (Baseline), Decision Tree, Random Forest, SVM, Gradient Boosting, MLP PyTorch |
| Modelo servido em produção | MLP PyTorch (`Linear→ReLU→Dropout→Linear(1)`) com pré-processamento via `ColumnTransformer` |
| Dataset | Telco Customer Churn (IBM) — `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Data do treino | 05/05/2026 |

---

## 2. Uso pretendido

- **Quem usa:** equipe de retenção; analistas de CRM; API síncrona e consumidores batch/orquestrados downstream.
- **Decisão suportada:** sinalizar clientes com risco elevado de cancelamento, para ação de retenção.
- **Fora de escopo:**
  - Estimativa do **valor monetário** de churn (LTV) — só prevê probabilidade.
  - **Causalidade**: o modelo não responde “porque é que vai cancelar”.
  - Multiclasse / regressão / séries temporais.

---

## 3. Dados

- **Fonte**: dataset público Telco Customer Churn (IBM, ~7.043 linhas).
- **Target**: coluna binária `Churn` (Yes/No → 1/0).
- **Features**: tabulares (contratuais, faturação) — engenharia adicional pela `ChurnFeatures` strategy.
- **Versão do CSV no run**: registado nos metrics (`fe_training_csv_basename`) e como dataset MLflow (`{objective}_fe_baseline_sample_csv`).
- **Split**: estratificado (80/20) com `random_state` fixo (`settings.random_state = 42`).
- **CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### Limitações dos dados

- **Desbalanceamento**: classe positiva (~26%) — mitigado parcialmente com `class_weight='balanced'` na LR baseline; tuning sklearn opera sobre split estratificado.
- **Cohort fixo**: dados são de um momento histórico — não captura efeitos sazonais recentes nem mudanças regulatórias após a recolha.
- **Cobertura geográfica**: clientes nos EUA — generalização para outros mercados não foi validada.

---

## 4. Arquitetura da MLP PyTorch MLP

| Componente | Valor |
|------------|-------|
| Tipo | Perceptron multicamadas para classificação binária |
| Camadas | `Linear(n_features, 64) → ReLU → Dropout → Linear(64, 32) → ReLU → Dropout → Linear(32, 1)` |
| Saída | 1 logit; probabilidade via `sigmoid(logit)` |
| Função de perda | `BCEWithLogitsLoss` |
| Optimizador | `AdamW (lr=1e-3, weight_decay=1e-5)` |
| Batching | `DataLoader(batch_size=64, shuffle=True)` |
| Early stopping | `patience=20` épocas sobre `val_loss` |
| Max épocas | 300 |
| Pré-processamento | `ColumnTransformer` (Standard scaler + OneHotEncoder) salvo no bundle (`*_preprocess.joblib`) |
| Validação | `train_test_split(stratify=y)` 15% para early stopping |
| Reprodutibilidade | `torch.manual_seed(42)` + `Generator` no `DataLoader` |

Bundle servido em produção (em `PATH_MODEL`):
- `pytorch_mlp_<obj>_<ts>.pt`
- `pytorch_mlp_<obj>_<ts>_preprocess.joblib`
- `pytorch_mlp_<obj>_<ts>_meta.json`

---

## 5. Métrica de optimização e baselines

- **Métrica de negócio**: maximizar **recall** (custo de FN > custo de FP — perder cliente que iria cancelar é mais caro do que oferecer retenção a quem não iria).
- **Métrica técnica de optimização** (CV): `recall` (configurável via Variable Airflow / form API).
- **Threshold de decisão (`decision_threshold`)**: configurável (`CLASSIFICATION_DECISION_THRESHOLD`). Default `0.5`. Em runs orientados a recall pode ser baixado para `0.25–0.4`.
- **Métricas reportadas** (todos os modelos, mesmo conjunto de teste): Accuracy, Precision, Recall, F1, ROC AUC.

> **Nota:** os modelos sklearn “pré-tuning” usam o mesmo split de teste; quando `USE_MLP_FOR_PREDICTION=false`, o sklearn “(tuned)” é o joblib servido. O **PyTorch MLP** corre em paralelo no mesmo run e é o que serve `/predict` quando `inference_backend='mlp'`.

---

## 6. Trade-off FP × FN (decisão de produto)

- **FP (Falso Positivo)**: cliente sinalizado como “vai cancelar”, mas que ficaria. Custo = ação de retenção desnecessária (desconto, ligação, brinde).
- **FN (Falso Negativo)**: cliente que vai cancelar mas não foi sinalizado. Custo = perda de receita recorrente + custo de aquisição de substituto.
- **Hipótese assumida**: custo de FN >> custo de FP (relação típica em telecom 5–10×).
- **Implicação**: optimizamos **recall**; o `decision_threshold` pode ser **reduzido** a um valor mais baixo (ex.: `0.30`) para apanhar mais positivos, ao custo de mais FP — preserva-se compatibilidade com `0.5` como default.
- O threshold efetivo fica gravado no run/bundle no momento do treino. Para mudar o comportamento em produção, treine e promova novo run com `decision_threshold` ajustado.
---

## 7. Limitações e cenários de falha

- **Desvio de distribuição (drift)**: o pipeline tem `scripts/maintenance/drift_report.py` (PSI). **Não há alerta em tempo real**; é offline / agendado. Recomenda-se rodar **semanalmente** ou após cada deploy.
- **Inputs fora do domínio**: `pandera` valida o schema de entrada (`tests/schemas/test_processor_schema.py` cobre o caso). Categorias novas em colunas categóricas: o `OneHotEncoder` foi treinado com `handle_unknown="ignore"` → não falha, mas pode reduzir qualidade silenciosamente. Mitigar com monitorização do PSI nessas colunas.
- **Classes ausentes em CV**: se o split do baseline ficar sem ambas as classes, métricas como `pr_auc` ficam `NaN` (já há logs de aviso); o pipeline continua, mas a comparação de campeão pode ser injusta.
- **Bundle MLP corrompido / parcial**: se faltar qualquer dos 3 ficheiros do bundle (`.pt`, `_preprocess.joblib`, `_meta.json`), o predict falha com erro explícito (`FileNotFoundError`). É preferível falhar cedo a inferir com pré-processamento errado.
- **Dependência de versão**: o `_preprocess.joblib` é `pickle` do sklearn. Mudanças major de versão do sklearn entre treino e serving podem invalidar o load — bloqueado em produção pela imagem Docker fixa.
- **Vieses potenciais**:
  - **Demográfico**: se o dataset sub-representar grupos (idade, dependentes), o modelo pode ter performance pior nesses subgrupos. Não foi feita análise por grupo nesta entrega — fica como dívida.
  - **Temporal**: cohort estático; pode degradar com mudanças de mercado.
  - **Selection bias**: dataset é de quem já tinha contrato — não cobre clientes que nunca chegaram a contratar.

---

## 8. Reprodutibilidade

- **Seeds fixos**: `RANDOM_STATE=42` em `train_test_split`, `StratifiedKFold`, `RandomForest`, `GradientBoosting`, `DecisionTree`, MLP (`torch.manual_seed`), `DataLoader.generator`.
- **Imagem Docker fixa**: dependências em `pyproject.toml` (Torch CPU, sklearn, MLflow); FastAPI servida sempre na mesma imagem (`processing_api:v1`) que treina e serve.
- **Tracking MLflow**: cada run grava params (incluindo `pytorch_mlp_*`), métricas (test/val), artefacto (`pytorch_mlp/` + bundle FE), e o CSV de treino.
- **Artefactos versionados**: tabela `pipeline_runs` + manifest no ZIP `fe_artifacts_<run_id>_<ts>.zip`.

---

## 9. Como reproduzir

```bash
# 1. .env
USE_MLP_FOR_PREDICTION=true   # MLP como backend de predict
RANDOM_STATE=42

# 2. subir
docker compose up -d

# 3. correr o DAG no Airflow (UI :8080) ou /v1/processor/admin/train/...
# 4. promover
curl -X POST http://localhost:8000/v1/processor/admin/promote -H "Authorization: Bearer ..."

# 5. inferência
curl -X POST http://localhost:8000/v1/processor/predict \
     -H "Content-Type: application/json" \
     -d @example_payload.json
```

Os artefatos do run ficam em `ml_data/<volume>/.../models/`, MLflow disponível pelo SQLite local (`mlflow.db`) ou via tracking server externo (`MLFLOW_TRACKING_URI`).

---

## 10. Plano de monitorização (resumo)

Detalhe completo em [`OBSERVABILIDADE_E_MANUTENCAO.md`](OBSERVABILIDADE_E_MANUTENCAO.md) e secção 7 do [`README.md`](../README.md). Pontos críticos:

| Indicador | Threshold | Acção |
|-----------|-----------|-------|
| PSI médio das features | < 0.10 | OK |
| PSI médio | 0.10–0.25 | aumentar frequência de monitorização |
| PSI médio | > 0.25 | retreinar com dados recentes + reavaliar drift |
| p95 latência `/predict` | < 300ms | OK; relatório em `latency_report.py` |
| Taxa de erro 5xx em `/predict` | < 1% | alerta + investigar antes de promover |

Periodicidade recomendada: após cada deploy + semanalmente com volume real.

---

## 11. Critério de promoção

Resumido (detalhe na seção “Regras de Decisão” do README):

1. O comparador de FE mantém `active=true` para o candidato vencedor por `cv_<optimization_metric>` quando a linhagem é comparável.
2. O promote exige exatamente um run FE `active=true`, `status='completed'`, `inference_backend` coerente com `USE_MLP_FOR_PREDICTION`, `objective=OBJECTIVE` e artefato existente.
3. Em produção, o run precisa ter `is_airflow_run=true`.
4. Gates de negócio como PSI < 0.10, margem mínima de métrica e aprovação humana ainda são controles externos ao endpoint.
5. Promote via `/v1/processor/admin/promote` (admin) → atualiza `DeployedModels`; em caso de problema, usar `/v1/processor/admin/rollback`.

---

## 12. Histórico de versões

| `pipeline_run_id` | `inference_backend` | best_model | Recall | F1 | ROC AUC | Notas |
|-------------------|---------------------|------------|--------|----|---------|-------|
| v1_teste | mlp | PyTorch MLP | 0.56 | 0.59 | 0.85 | _entrega Tech Challenge_ |

---

## 13. Contatos / Responsáveis

- Equipe: Alexandre Lucena; Eliane Karasawa; Gabriel Drumond; Marcus de Carvalho; Matheus Pessoa
- Repo: `[<url>](https://github.com/Heroi211/Machine-Learning)`

---

## Anexos

- Tabela comparativa completa: `fe_export_<obj>_<ts>/model_comparison_full.md` (incluída no ZIP do FE).
- CSV equivalente: `fe_export_<obj>_<ts>/model_comparison_full.csv`.
- Resumo MLP: `fe_export_<obj>_<ts>/pytorch_mvp_summary.csv`.
- MLflow: experiments `<objective>_baseline` e `<objective>_feature_engineering`.
