# Relatório analítico — entrada do domínio Churn

Documento de avaliação (visão de mentor) baseado no código atual do repositório, com foco em **recall** como métrica de otimização e na decisão **Random Forest vs XGBoost**.

---

## 1. Estado atual do projeto (o que já favorece churn)

| Aspeto | Situação no código |
|--------|---------------------|
| **Enum de domínio** | `MLDomain` em `schemas/processor_schemas.py` já inclui `churn`. Swagger e forms aceitam o valor. |
| **Labels para Baseline / gráficos** | `CLASS_LABELS["churn"] = ("Não Churn", "Churn")` em `feature_strategies/__init__.py`. |
| **Métrica `recall`** | Já suportada de ponta a ponta: `ALLOWED_OPTIMIZATION_METRICS` em `fe_model_selection.py` inclui `recall`; `train_models` ordena pelo nome de coluna PT (`Recall`); `tune()` usa `sklearn_scoring_parameter` no CV. |
| **Pipeline FE genérico** | `FeatureEngineering` é agnóstico ao domínio depois de instanciada com uma `FeatureStrategy`; basta registar a strategy. |

Ou seja: a **infraestrutura de métrica e de domínio no API** está preparada; o que falta é **ML por domínio** (strategy + predição) e, se quiseres XGB, **extensão do catálogo de modelos**.

---

## 2. Lacunas obrigatórias antes de treinar churn com FE

1. **`STRATEGY_REGISTRY`** — Só existe `HeartDiseaseFeatures`. É necessária uma classe **`ChurnFeatures`** (ou nome equivalente) que implemente `FeatureStrategy`: `required_columns`, `build`, `created_features`, `validate`.
2. **Registo** — `"churn": ChurnFeatures` em `feature_strategies/__init__.py`.
3. **Dataset** — CSV com contrato do projeto: **última coluna `target`**, binário; colunas alinhadas ao que a strategy exige (ex. Telco: `tenure`, charges, serviços, etc., após normalizar nomes para minúsculas no `load_data` do FE).
4. **Predição (`POST /predict`)** — `PredictRequest.features` é hoje **só** `HeartDiseaseFeaturesInput`. Para churn é necessário **Union discriminada por `domain`** ou schema `ChurnFeaturesInput` + validação condicional; caso contrário a API rejeita corpos válidos de churn.

Sem os pontos 1–3 o FE com `objective=churn` falha; sem o ponto 4 o serving não fecha o ciclo.

---

## 3. Recall como métrica principal (churn)

**Encaixe com o negócio:** em churn, maximizar **recall** da classe positiva (churn) costuma alinhar-se a “não deixar escapar clientes que vão sair”, aceitando mais falsos alarmes (custo operacional em campanhas).

**Encaixe com o código:**

- Passar `optimization_metric=recall` nas rotas/API/Airflow `conf` já direciona ranking em `train_models` e o scoring interno do tuning.
- **Limitação:** `recall_score` no sklearn usa por defeito a classe **positiva como `1`**. Garante que o teu `target` no CSV está codificado como **1 = churn** (e 0 = não churn). Se estiver invertido, o recall otimizado será o da classe errada.
- **Desbalanceamento:** churn costuma ser minoritário. O projeto usa `class_weight='balanced'` no Baseline; no FE os modelos sklearn por defeito **não** aplicam pesos — para recall em classes desbalanceadas convém avaliar **`class_weight='balanced'`** em RF/DT e, no XGB, **`scale_pos_weight`** (ver secção 5).

---

## 4. Random Forest vs XGBoost — o que o código faz hoje

**Situação real no repositório:**

- **Random Forest:** presente em `train_models` e em `fe_hyperparameter_tuning.py` (tuning com `n_estimators`, `max_depth`, `min_samples_*`, `max_features`).
- **“Gradient Boosting”** no projeto é **`sklearn.ensemble.GradientBoostingClassifier`**, **não** XGBoost.
- **XGBoost** (`xgboost.XGBClassifier`): **não** aparece em `requirements.txt` nem em `train_models`; não há ramo em `param_distributions_for`.

**Implicação:** escolher entre **RF** e **XGB** implica, para XGB, **nova dependência** + alterações em cadeia:

- `requirements.txt` / imagem Docker.
- `train_models`: novo entrada no dicionário `model_configs` (nome amigável, ex. `"XGBoost"`).
- `fe_hyperparameter_tuning.py`: `build_fresh_tuning_pipeline` + `param_distributions_for` para hiperparâmetros típicos (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `scale_pos_weight`, etc.).
- Garantir **`predict_proba`** para ROC AUC e para o fluxo atual de métricas.

**Comparação rápida (conceitual):**

|  | Random Forest | XGBoost |
|--|---------------|---------|
| Interpretação / robustez | Menos sensível a escala; pouco tuning | Ganha com tuning; sensível a `learning_rate` / profundidade |
| Dados tabulares churn | Muito usado, baseline forte | Muito usado em competições; bom com interações |
| Parâmetros distintos | Árvores paralelas (`n_estimators`, `max_depth`, amostragem de features) | Boosting sequencial (`eta`, `max_depth`, regularização, `scale_pos_weight`) |
| Custo de integração no teu repo | Já integrado | Médio — novos ficheiros e testes |

**Recomendação pragmática:** começar com **RF** já suportado, `optimization_metric=recall`, e opcionalmente ativar **`class_weight='balanced'`** no estimador para alinhar ao desbalanceamento; depois adicionar **XGB** como quinto modelo se o ganho de recall no teu holdout justificar o custo de manutenção.

---

## 5. Hiperparâmetros e recall

- **RF (já no projeto):** o `ParameterSampler` atual explora amplitude razoável; para churn, validar se `max_features` e profundidade não underfittam com poucas features.
- **XGB (futuro):** `scale_pos_weight ≈ neg/pos` é central quando o target é desbalanceado e a métrica é recall da classe minoritária; cruzar com early stopping opcional para não sobreajustar.

O loop `tune()` usa **validação cruzada estratificada** e a métrica vinda de `sklearn_scoring_parameter` — para recall, confirma que o **positivo** é mesmo churn (codificação `y`).

---

## 6. Ordem de implementação sugerida

1. Definir schema de colunas do dataset churn e implementar **`ChurnFeatures`** + registo no registry.
2. Correr **Baseline** + **FE** em dev com `optimization_metric=recall` e validar métricas no MLflow.
3. Estender **`PredictRequest`** para features de churn (Union ou segundo modelo).
4. Testar **promote** + **predict** end-to-end.
5. (Opcional) Adicionar **XGBoost** ao comparativo e ao tuning.
6. (Opcional) Persistência de **`pipeline_runs`** a partir do DAG Airflow, se o treino oficial for só orquestrado.

---

## 7. Riscos e pontos de atenção

- **API de predição** é o maior bloqueio arquitetural atual para “churn completo” — sem novo schema, o enum `churn` existe mas o body continua heart-only.
- **DAG Airflow** valida domínio via `STRATEGY_REGISTRY`; após registar churn, a validação passa, mas **persistência de runs na BD** pelo DAG continua um débito (ver `DOCUMENTACAO_SOLUCAO.md`).
- **Dozzle vs Airflow:** métricas finas ficam nos logs da task no Airflow UI; não substituir por apenas logs do contentor.

---

*Elaborado com base em `feature_engineering.py`, `fe_model_selection.py`, `fe_hyperparameter_tuning.py`, `processor_schemas.py`, `processor_service.py` e `feature_strategies/`.*
