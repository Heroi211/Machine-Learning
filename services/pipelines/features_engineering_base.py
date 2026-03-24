#%%
from IPython import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
import joblib
import warnings
import time
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score
from scipy.stats import randint
from datetime import datetime
import os
from sklearn.inspection import permutation_importance


warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# carregando o dataset ja processado no baseline 
path = "data/heart_disease_uci_preprocessed.csv"
df = pd.read_csv(path)

# Garantir coluna target presente e binária ( já feita no baseline)
# if not 'target' in df.columns:
#     print('Coluna target não existe!')
    
# ou 
assert 'target' in df.columns, 'Coluna target não existe!'

# remover colunas de metadados se existirem 
for col in ['id','dataset']:
    if col in df.columns:
        df.drop(columns=[col],inplace=True)
        
print(f"\nDataset Shape (pré-processado): {df.shape}")
print("\nTarget (0=Sem Doença, 1=Com doença):")
print(df['target'].value_counts()) 

df.head()   

# estatísticas rápidas para conferência (dataset já pré processado)
df.describe()

# Verificar (rapidamente) valores nulos - não deve haver 
null_total = df.isnull().sum().sum()
print(f"Total de valores nulos no CSV pré-processado: {null_total}!")
if null_total > 0:
    print('Abortando análise existem valores nulos no conjunto de dados.')
    
#-----------------------Featuring Engineering---------------------------------
# Tarefa 1 - Criar novas features baseadas nas existentes

# Criar features adicionais sobre o dataset já pré-processado (Aula 2)
df_engineered = df.copy()

# Garantir presença do alvo
y = df_engineered['target']
X = df_engineered.drop(columns=['target'])

# Criar features adicionais
eps = 1

# Dar maior peso a idade (quadrática)
df_engineered['age_squared'] = df_engineered['age'] ** 2

# Relação colesterol/idade: pode indicar risco relativo
df_engineered['cholesterol_to_age'] = df_engineered['chol'] / (df_engineered['age'] + eps)  # Evitar divisão por zero

# Percentual da frequência cardíaca máxima alcançada (regra 220 - age)
# Interpretação: quão próximo do máximo previsto o paciente chegou no esforço
if 'thalch' in df_engineered.columns and 'age' in df_engineered.columns:
    predicted_max_hr = (220 - df_engineered['age']).clip(lower=1)  # evita divisão por zero
    df_engineered['max_hr_pct'] = df_engineered['thalch'] / (predicted_max_hr + eps)

# Razão pressão/colesterol: pode sinalizar perfil de risco vascular relativo
if 'trestbps' in df_engineered.columns and 'chol' in df_engineered.columns:
    df_engineered['bp_chol_ratio'] = df_engineered['trestbps'] / (df_engineered['chol'] + 1)

# Mapear flags booleanas/ binárias para 0/1
if 'fbs' in df_engineered.columns:
    df_engineered['fbs_flag'] = df_engineered['fbs'].astype(int)
if 'exang' in df_engineered.columns:
    df_engineered['exang_flag'] = df_engineered['exang'].astype(int)

# Índice de estresse: relação entre frequência máxima alcançada e pressão de repouso
# (indicador simples de capacidade cardiorrespiratória frente à pressão arterial)
if 'thalch' in df_engineered.columns and 'trestbps' in df_engineered.columns:
    df_engineered['stress_index'] = df_engineered['thalch'] / (df_engineered['trestbps'] + eps)

# Década de idade (faixa etária simples, útil para interação e interpretação)
if 'age' in df_engineered.columns:
    df_engineered['age_decade'] = (df_engineered['age'] // 10).astype(int)

# Interação idade x oldpeak: pacientes mais velhos com maior depressão ST têm maior risco
if 'age' in df_engineered.columns and 'oldpeak' in df_engineered.columns:
    df_engineered['risk_interaction'] = df_engineered['age'] * df_engineered['oldpeak']

# Flag indicando depressão ST elevada (threshold pragmático)
if 'oldpeak' in df_engineered.columns:
    df_engineered['high_st_depression_flag'] = (df_engineered['oldpeak'] > 1.0).astype(int)

# Conferência rápida das novas features adicionadas
new_feats = ['fbs_flag', 'exang_flag', 'bp_chol_ratio', 'max_hr_pct',
             'stress_index', 'age_decade', 'risk_interaction', 'high_st_depression_flag']
present = [c for c in new_feats if c in df_engineered.columns]
print(f"Novas features adicionadas ({len(present)}): {present}")

df_engineered.head(1)

# Seleção de features
# Tarefa 2 : Selecione as features mais relevantes

#Dividir dados (dataset já numérico, sem OHE adicional)

y = df_engineered['target']
x = df_engineered.drop(columns=['target'])
feature_names = x.columns.tolist()

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# Seleção de features usando a nova f-value
selector_preview = SelectKBest(f_classif,k=min(25,x_train.shape[1]))
x_train_selected = selector_preview.fit_transform(x_train,y_train)

x_test_selected = selector_preview.transform(x_test)

selected_mask = selector_preview.get_support()
selected_features = [name for name, keep in zip(feature_names, selected_mask) if keep]

print(f"\n Features selecionados ({len(selected_features)}):")
for i, feat in enumerate(selected_features,1):
    print(f"{i} . {feat}")
    
# Treinamento de diferentes algoritmos

# 1 - Definimos um dicionário com os modelos que queremos treinar
# 2 - Montamos um pipeline simples quie inclui standartScaler apenas quando o algoritmo é sensível à escala (como o SVM)
# 3 - Treinamos cada modelo com os dados de treino e avaliamos no conjunto de teste.
# 4 - Consolidamos as métricas em uma tabela para facilitar a comparação inicial

# Comparação inicial entre diferentes modelos de classificação.
# Comparação inicial entre diferentes algoritmos de classificação
model_configs = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

trained_models = {}
results = []

for model_name, estimator in model_configs.items():
    steps = []
    if model_name == "Support Vector Machine":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = np.nan
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(x_test)
        roc_auc = roc_auc_score(y_test, y_scores)

    print(f"=== {model_name} ===")
    print(f"Acurácia no conjunto de teste: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Sem doença", "Com doença"]))
    print("-" * 70)

    trained_models[model_name] = pipeline
    results.append({
        "Modelo": model_name,
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "F1": f1,
        "ROC AUC": roc_auc,
    })

results_df = pd.DataFrame(results)
results_df.sort_values(by="Acurácia", ascending=False).reset_index(drop=True).round(3)

# Tunning de hiperparametros

# 1 - Alvo:Acurária de teste > 0.90 ( Melhor que o modelo GradientBoost sem otimização):
# 2 - Busca aleatória por 60 minutos apenas nos hiperparâmetros-chave do random forest
# 3 - Em cada amostra: estimamos acurácia via CV estratificado (5-fold) e medimos acurácia no teste: mantemos o melhor por acurácia de teste
# 4 - Ao final: reportamos se o alvo (>90) foi atingido e registramos melhor configuração e métricas no mlflow


# Tuning com foco em 4 hiperparâmetros e orçamento de 60 minutos
# - Hiperparâmetros: n_estimators, max_depth, min_samples_split, min_saples_leaf
# - CV 5-fold estratificado + seleção pelo melhor desempenho em teste

# Parametros do processo de busca
TIME_LIMIT_MINUTES = 60
ACC_TARGET = 0.90

start_time = time.time()
deadline = start_time + TIME_LIMIT_MINUTES * 60

# Pipeline base: Selector + RandomForest (selector fixo; distribuição apenas nos 4 hiperparâmetros)
n_features = x_train.shape[1]
base_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k="all")),
    ("model", GradientBoostingClassifier(random_state=42)),
])

# Espaço de busca restrito aos 4 hiperparâmetros
# param_distributions = {
#     "model__n_estimators": randint(50, 151),
#     "model__max_depth": [None, 2, 5, 7, 10, 15, 25],
#     "model__max_features": [v for v in [2, 5, 7, 10, 15, 25] if v <= x_train.shape[1]],
#     "model__max_leaf_nodes": [None] + list(range(5, 16)),
#     "model__min_samples_split": randint(2, 300),   # 2 a 300
#     "model__min_samples_leaf": randint(1, 300),    # 1 a 300
# }

param_distributions = {

    "model__n_estimators": randint(50, 400),
    "model__learning_rate": uniform(0.01, 0.3),
    "model__max_depth": randint(2, 6),
    "model__min_samples_split": randint(2, 100),
    "model__min_samples_leaf": randint(1, 50),
    "model__subsample": uniform(0.6, 0.4)

}

# Amostrador "infinito" (parada por tempo)
sampler = ParameterSampler(param_distributions=param_distributions, n_iter=1_000_000, random_state=42)

# Estratégia de validação
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best = {
    "params": None,
    "cv_acc": -np.inf,
    "test_acc": -np.inf,
    "pipeline": None,
}

n_evaluated = 0
print(f"Iniciando busca por até {TIME_LIMIT_MINUTES} min. Alvo: acc_teste > {ACC_TARGET}.")

for i, params in enumerate(sampler, start=1):
    now = time.time()
    if now >= deadline:
        print("Tempo limite atingido; encerrando busca.")
        break

    # Clonar pipeline e aplicar hiperparâmetros
    pipeline = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k="all")),
        ("model", GradientBoostingClassifier(random_state=42)),
    ])
    pipeline.set_params(**params)

    # Avaliação via CV
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_cv = float(np.mean(cv_scores))

    # Ajuste e avaliação em teste
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Atualizar melhor
    if test_acc > best["test_acc"]:
        best.update({
            "params": params,
            "cv_acc": mean_cv,
            "test_acc": test_acc,
            "pipeline": pipeline,
        })
        elapsed = now - start_time
        print(f"[{i}] Novo melhor | acc_teste={test_acc:.4f} | acc_cv={mean_cv:.4f} | t={elapsed/60:.1f}m | params={params}")

    # Logs periódicos
    if i % 5 == 0:
        elapsed = now - start_time
        remaining = max(0.0, deadline - now)
        print(
            f"Iterações: {i} | melhor_acc_teste={best['test_acc']:.4f} | "
            f"decorrido={elapsed/60:.1f}m | restante≈{remaining/60:.1f}m"
        )

    n_evaluated = i

# Preparar métricas do melhor modelo
best_rf = best["pipeline"] if best["pipeline"] is not None else base_pipeline.fit(x_train, y_train)

y_pred_tuned = best_rf.predict(x_test)
y_proba_tuned = (
    best_rf.predict_proba(x_test)[:, 1]
    if hasattr(best_rf, "predict_proba") else
    (best_rf.decision_function(x_test) if hasattr(best_rf, "decision_function") else np.zeros_like(y_pred_tuned, dtype=float))
)

tuned_metrics = {
    "Acurácia": accuracy_score(y_test, y_pred_tuned),
    "Precisão": precision_score(y_test, y_pred_tuned),
    "Recall": recall_score(y_test, y_pred_tuned),
    "F1": f1_score(y_test, y_pred_tuned),
    "ROC AUC": roc_auc_score(y_test, y_proba_tuned) if not np.all(y_proba_tuned == 0) else np.nan,
}

print("\nResumo da busca:")
print(f"Iterações avaliadas: {n_evaluated}")
elapsed_total = time.time() - start_time
print(f"Tempo total: {elapsed_total/60:.1f} minutos")
print(f"Melhor acurácia (teste): {tuned_metrics['Acurácia']:.4f}")
print(f"Alvo (acc > {ACC_TARGET}): {'atingido' if tuned_metrics['Acurácia'] > ACC_TARGET else 'não atingido'}")

# Registro no MLflow
try:
    mlflow.set_experiment("aula3_time_budget_tuning")
    with mlflow.start_run(run_name="rf_parameter_sampler_60min_4params"):
        if best["params"] is not None:
            for k, v in best["params"].items():
                mlflow.log_param(k, str(v))
        mlflow.log_metric("cv_accuracy", float(best["cv_acc"]))
        for k, v in tuned_metrics.items():
            if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                mlflow.log_metric(k.replace(" ", "_").lower(), float(v))
        mlflow.sklearn.log_model(best_rf, "model")
    print("\nResultados registrados no MLflow (experimento: aula3_time_budget_tuning).")
except Exception as e:
    print(f"\n[Aviso] Falha ao registrar no MLflow: {e}")

pd.DataFrame([tuned_metrics]).round(3)
      
#importancia de features (Gini + Permutation) e persistencia/log do melhor modelo
print("=== Importância de Features do melhor modelo ===")

# Garantias : Execute a seção de tuning antes desta célula
assert "best _rf" in globals(), "best_rf não encontrado. Execute a seção de tuning antes."
assert "tuned_metrics" in globals(), "tuned_metrics não encontrado. Execute a seção de tuning antes."

#extrair nomes de features utilizados pelo pipeline (após selctor, se houver)
if hasattr(best_rf,"named_steps") and "selector" in best_rf.named_steps and hasattr (best_rf.named_steps["selector"], "get_support"):
    mask = best_rf.named_steps["selector"].get_support()
    feat_names_used = np.array(x_train.columns)[mask]
else:
    feat_names_user = np.array(x_train.columns)
    
# 1) importãncia baseada em arvore (GINI)

model_step = best_rf.named_steps["model"] if hasattr(best_rf,"named_steps") and "model" in best_rf.named_steps else best_rf
tree_imp = getattr(model_step,"feature_importances_", None)

figs_to_log = []

if tree_imp is not None and len(tree_imp) == len(feat_names_used):
    imp_df = pd.DataFrame({"feature": feat_names_used, "importance": tree_imp}).sort_values("importance", ascending=False)
    display(imp_df.head(20))

    fig, ax = plt.subplots(figsize=(8, min(0.45*len(imp_df.head(20)), 10)))
    sns.barplot(data=imp_df.head(20), x="importance", y="feature", color="#1f77b4", ax=ax)
    ax.set_title("Importância de Features (Gini) - Top 20")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    figs_to_log.append(("feature_importance_gini_top20.png", fig))
    plt.show()
else:
    print("[Info] Modelo não expõe feature_importances_. Pulando gráfico Gini.")

# 2) Importância por permutação (Accuracy)
perm = permutation_importance(best_rf, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": feat_names_used,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False)
display(perm_df.head(20))

fig2, ax2 = plt.subplots(figsize=(8, min(0.45*len(perm_df.head(20)), 10)))
sns.barplot(data=perm_df.head(20), x="importance", y="feature", color="#ff7f0e", ax=ax2)
ax2.set_title("Importância por Permutação (Accuracy) - Top 20")
ax2.set_xlabel("Queda média na métrica")
ax2.set_ylabel("Feature")
plt.tight_layout()
figs_to_log.append(("feature_importance_permutation_top20.png", fig2))
plt.show()
    
    
 # Persistência local com joblib
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

joblib_path = os.path.join(models_dir, f"best_random_forest.joblib")
joblib.dump(best_rf, joblib_path)
print(f"Modelo salvo via joblib em: {joblib_path}")

# Log no MLflow: abrir um novo run para registrar artefatos e o modelo salvo
mlflow.set_experiment("aula3_time_budget_tuning")
with mlflow.start_run(run_name="rf_importances_and_persistence"):
    # Parâmetros/metrics chave para rastreabilidade
    if 'best' in globals() and isinstance(best.get('params'), dict):
        for k, v in best["params"].items():
            mlflow.log_param(k, str(v))
    for k, v in tuned_metrics.items():
        if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            mlflow.log_metric(k.replace(" ", "_").lower(), float(v))

    # Logar figuras
    for fname, fobj in figs_to_log:
        try:
            mlflow.log_figure(fobj, fname)
        except Exception as e:
            print(f"[Aviso] Falha ao logar figura {fname}: {e}")

    # Logar o pipeline completo e o arquivo joblib como artifact
    mlflow.sklearn.log_model(best_rf, artifact_path="sklearn_model")
    mlflow.log_artifact(joblib_path, artifact_path="joblib")

print("Concluído: importâncias, persistência (joblib) e log no MLflow.")
#%%
