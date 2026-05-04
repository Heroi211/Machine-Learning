# Implementação da MLP (PyTorch)

Documentação técnica da rede neural usada como modelo principal do desafio.
Tudo aqui descreve o que **está no código** — sem teoria genérica.

> Para a visão geral do projeto, ver [`../README.md`](../README.md). Para limitações e cenários de falha do produto, ver [`MODEL_CARD.md`](MODEL_CARD.md).

---

## Sumário

1. [Conceito em uma frase](#1-conceito-em-uma-frase)
2. [Onde está o código](#2-onde-está-o-código)
3. [Arquitetura camada a camada](#3-arquitetura-camada-a-camada)
4. [Loop de treino passo a passo](#4-loop-de-treino-passo-a-passo)
5. [Hiperparâmetros](#5-hiperparâmetros)
6. [Pré-processamento MLP vs sklearn](#6-pré-processamento-mlp-vs-sklearn)
7. [Bundle de inferência (3 arquivos)](#7-bundle-de-inferência-3-arquivos)
8. [Inferência em produção](#8-inferência-em-produção)
9. [Reprodutibilidade](#9-reprodutibilidade)
10. [Falhas silenciosas](#10-falhas-silenciosas)
11. [Comparação MLP vs sklearn no `/predict`](#11-comparação-mlp-vs-sklearn-no-predict)

---

## 1. Conceito em uma frase

Uma **MLP (Multi-Layer Perceptron)** é uma rede neural *feedforward* que recebe um vetor numérico de features e produz **um único número** (o "logit"). Aplicando `sigmoid` a esse logit obtém-se a probabilidade de churn (0–1).

---

## 2. Onde está o código

| Responsabilidade | Arquivo |
|------------------|---------|
| Definição da rede + treino + avaliação | `src/services/pipelines/mlp_torch_tabular.py` |
| Carregamento do bundle + predição em produção | `src/services/pipelines/mlp_inference.py` |
| Integração no pipeline FE | `FeatureEngineering._run_mlp_torch_mvp()` em `src/services/pipelines/feature_engineering.py` |

---

## 3. Arquitetura camada a camada

Classe `_MLPBinary` em `mlp_torch_tabular.py`. Defaults: `hidden_dims=(64, 32)`, `dropout=0.0`. Para `N` features de entrada (após `OneHotEncoder` + `StandardScaler`):

| # | Camada | Forma | O que faz |
|---|--------|-------|-----------|
| 1 | `Linear(N, 64)` | N → 64 | 64 combinações lineares das features (peso por feature + bias). |
| 2 | `ReLU()` | 64 → 64 | Não-linearidade — zera negativos, mantém positivos. Sem ela, a rede colapsaria em uma regressão linear. |
| 3 | `Dropout(p)` | 64 → 64 | Só no treino; desliga aleatoriamente uma fração `p` dos neurônios. Identidade quando `p=0`. |
| 4 | `Linear(64, 32)` | 64 → 32 | Reduz dimensão; aprende combinações de alto nível. |
| 5 | `ReLU()` | 32 → 32 | Não-linearidade. |
| 6 | `Dropout(p)` | 32 → 32 | Idem item 3. |
| 7 | `Linear(32, 1)` | 32 → 1 | Saída. **Sem ativação** — produz o logit. |

Forward: `model(x).squeeze(-1)` → vetor 1D de logits.

### Por que **não** existe `Sigmoid` na última camada?

A `BCEWithLogitsLoss` (loss usada no treino) aplica `sigmoid + binary cross-entropy` numa única operação **numericamente estável** (evita overflow/underflow em logits extremos). É a forma recomendada pela documentação do PyTorch para classificação binária.

---

## 4. Loop de treino passo a passo

Implementado em `train_eval_mlp_binary_tabular`:

1. **Sementes determinísticas**:
   - `torch.manual_seed(random_state)` antes de instanciar a rede.
   - `torch.cuda.manual_seed_all(random_state)` se houver GPU.
   - `Generator(seed=random_state)` separado, passado ao `DataLoader(shuffle=True)` para fixar a ordem dos batches.
2. **Tensores**: `X` em `float32` denso (sparse → `.toarray()`); `y` em `float32` (BCE exige float). Movidos para `device` (CUDA se disponível, senão CPU).
3. **DataLoader**: `TensorDataset(X_train, y_train)`, `batch_size=min(64, len(X_train))`, `shuffle=True`, `drop_last=False`.
4. **Para cada época** (até `max_epochs=300`):
   - `model.train()` — ativa Dropout (se houver).
   - **Para cada batch**:
     - `optimizer.zero_grad(set_to_none=True)` — limpa gradientes.
     - `logits = model(xb)` — forward pass.
     - `loss = BCEWithLogitsLoss(logits, yb)` — sigmoid + BCE estável.
     - `loss.backward()` — backpropagation (calcula `∂loss/∂peso`).
     - `optimizer.step()` — `AdamW` atualiza pesos (`lr=1e-3`, `weight_decay=1e-5`).
   - **Fim da época**:
     - `model.eval()` (desliga Dropout) + `torch.no_grad()` (sem gradientes).
     - Calcula `val_loss` no conjunto de validação inteiro.
5. **Early stopping** (`patience=20`):
   - Se `val_loss < melhor_anterior - 1e-6` → salva `state_dict` como `best_state`, zera o contador.
   - Caso contrário → decrementa. Se chegar a 0 → para.
6. **Restauração**: ao final, carrega `best_state` (não os pesos da última época).

### Por que `AdamW` e não SGD?

`AdamW` é **adaptativo** — ajusta a taxa de aprendizado por parâmetro automaticamente — e separa `weight_decay` da atualização de gradiente (mais correto matematicamente que o `Adam` clássico). É a escolha padrão moderna para tabular pequeno/médio.

### Avaliação interna (durante o treino)

Após carregar `best_state`:
- `logits_test = model(X_test)` (sem gradientes).
- `proba_test = 1 / (1 + exp(-logits_test))` — sigmoid manual em NumPy.
- `pred_test = (proba_test >= 0.5).astype(int64)` — **threshold fixo de 0.5**.
- Métricas: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (se y_true tiver ambas as classes).

> **Atenção**: o `0.5` aqui é só para reportar métricas internas no MLflow / `pytorch_mvp_summary.csv`. O threshold de produção é o da env `CLASSIFICATION_DECISION_THRESHOLD`, gravado no `_meta.json` do bundle e usado por `predict_with_mlp`.

---

## 5. Hiperparâmetros

Defaults em `FeatureEngineering.__init__`:

| Parâmetro | Default | Significado prático |
|-----------|---------|---------------------|
| `mlp_hidden_dims` | `(64, 32)` | Duas camadas ocultas — capacidade moderada para o tamanho do Telco. |
| `mlp_dropout` | `0.0` | Sem dropout. Aumentar (0.2–0.3) ajuda em datasets ruidosos. |
| `mlp_batch_size` | `64` | Compromisso típico para ~7k linhas. |
| `mlp_lr` | `1e-3` | Taxa de aprendizado padrão do `AdamW`. |
| `mlp_weight_decay` | `1e-5` | L2 regularization "leve" embutida. |
| `mlp_max_epochs` | `300` | Teto duro — early stopping costuma parar muito antes. |
| `mlp_early_stopping_patience` | `20` | Tolera 20 épocas sem melhora antes de parar. |
| `mlp_val_fraction` | `0.15` | 15% do treino reservado para `val_loss` (estratificado). |

---

## 6. Pré-processamento MLP vs sklearn

| Etapa | Pipelines sklearn (DT/RF/SVM/GB) | MLP PyTorch |
|-------|----------------------------------|-------------|
| `ColumnTransformer` (imputer + scaler + OHE) | ✅ | ✅ (mesmo objeto, mesmo `fit_transform`) |
| `VarianceThreshold(0.0)` | ✅ | ❌ |
| `SelectKBest(f_classif, k=25)` | ✅ | ❌ |
| Modelo final | DT / RF / SVM / GB | `_MLPBinary` |

### Por que a MLP não usa `SelectKBest`?

- A própria primeira camada (`Linear(N, 64)`) aprende a "anular" features irrelevantes via pesos próximos de zero.
- Forçaria a MLP a competir em desigualdade com sklearn (entrada menor).

**Trade-off explícito**: na tabela `model_comparison_full.csv`, a MLP recebe **todas as colunas pós-OHE** enquanto Random Forest/Gradient Boosting recebem apenas as 25 selecionadas. Comparações entre eles não são "modelo A vs modelo B" estritamente igual.

---

## 7. Bundle de inferência (3 arquivos)

Cada run FE com MLP gera, em `PATH_MODEL`, três arquivos com o prefixo `pytorch_mlp_<objective>_<run_ts>`:

| Arquivo | Conteúdo | Para quê serve |
|---------|----------|----------------|
| `<prefix>.pt` | `state_dict` (apenas pesos) | Tensores aprendidos. Não contém a classe `_MLPBinary`. |
| `<prefix>_preprocess.joblib` | `ColumnTransformer` ajustado | Aplica a mesma normalização/OHE em produção. |
| `<prefix>_meta.json` | `hidden_dims`, `dropout`, `n_features_in`, `decision_threshold`, `feature_columns_in_order`, `feature_groups`, `best_epoch`, `best_val_loss`, `metrics_test/val`, `torch_version` | Recria a arquitetura exatamente + contrato de colunas + threshold de produção. |

### Por que 3 arquivos separados?

- **Separação de responsabilidades**: pesos (PyTorch) ≠ pré-processamento (sklearn) ≠ contrato (JSON legível).
- O `state_dict` puro é menor e mais portátil que serializar a classe inteira via `pickle`.
- O `_meta.json` é **inspecionável por humanos** — útil para auditoria e debug.

---

## 8. Inferência em produção

Quando `POST /predict` chega para um run com `inference_backend="mlp"`:

1. `processor_service._prepare_prediction_features` aplica `ChurnFeatures.build`, com `monthly_median` **reidratado** de `pipeline_runs.metrics["strategy_monthly_charges_median"]`.
2. `load_mlp_bundle(prefix)`:
   - Lê os 3 arquivos.
   - Reconstrói `_MLPBinary(n_features=meta.n_features_in, hidden_dims=meta.hidden_dims, dropout=meta.dropout)`.
   - `model.load_state_dict(torch.load(<prefix>.pt, map_location="cpu"))`.
   - `model.eval()` — desliga Dropout para inferência.
3. `predict_with_mlp(bundle, df_input, threshold=None)`:
   - Alinha colunas pela ordem em `meta.feature_columns_in_order` — coluna ausente → `ValueError` (HTTP 400).
   - `bundle.preprocess.transform(df)` → matriz `float32` densa.
   - `torch.no_grad()` + `model(tensor)` → logit.
   - `proba = 1 / (1 + exp(-logit))` → P(churn).
   - `label = 1 if proba >= threshold else 0` (threshold = `bundle.decision_threshold` se não houver override).

---

## 9. Reprodutibilidade

| Ponto | Mecanismo |
|-------|-----------|
| Pesos iniciais | `torch.manual_seed(random_state)` antes de instanciar `_MLPBinary`. |
| GPU (se houver) | `torch.cuda.manual_seed_all(random_state)`. |
| Ordem de batches no `DataLoader` | `Generator(seed=random_state)`. |
| Split treino/val do MLP | `train_test_split(stratify=y, random_state=42)`. |
| Sementes do sklearn (CV, modelos) | `random_state=42` em todos os pontos. |

`random_state` vem de `settings.random_state` (env `RANDOM_STATE=42` no `.env_example`).

---

## 10. Falhas silenciosas

Em `_run_mlp_torch_mvp`:

- `enable_mlp_torch=False` → passo é pulado com log; o run FE continua só com sklearn.
- `import torch` falha → log `"MLP PyTorch indisponível (import)"` + passo pulado.
- Treino lança exceção → log com `exc_info=True` + `mlp_torch_result = None` + sklearn segue normal.

Em qualquer caso acima, o run **não pode** ser promovido com `USE_MLP_FOR_PREDICTION=true` (não haverá bundle MLP). O `promote` falha explicitamente exigindo coerência entre `inference_backend` do run e a env.

---

## 11. Comparação MLP vs sklearn no `/predict`

| Aspeto | sklearn (joblib) | MLP (PyTorch) |
|--------|------------------|---------------|
| Artefato em disco | 1 arquivo `.joblib` | Bundle de 3 arquivos com mesmo prefixo. |
| Como o run sinaliza | `pipeline_runs.inference_backend = "sklearn"` | `pipeline_runs.inference_backend = "mlp"` + `metrics.mlp_artifact_prefix` |
| Decisão final | `model.predict(X)` (threshold interno ~0.5) | `proba >= decision_threshold` (vem do `_meta.json`) |
| Customização de threshold | Indireta (precisa recodar serving) | Direta — basta mudar a env e re-treinar |
| Suporte a GPU | Não | Sim (auto-detect; usa CPU se não houver GPU) |
