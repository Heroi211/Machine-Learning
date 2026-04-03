"""
Pipelines e distribuições de hiperparâmetros para o tuning pós-comparação de modelos no FE.

Cada nome de modelo coincide com os usados em `train_models` (FeatureEngineering).
"""

from __future__ import annotations

from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Nomes exatamente como em train_models
DECISION_TREE = "Decision Tree"
RANDOM_FOREST = "Random Forest"
SVM = "SVM"
GRADIENT_BOOSTING = "Gradient Boosting"

SUPPORTED_MODELS = frozenset({DECISION_TREE, RANDOM_FOREST, SVM, GRADIENT_BOOSTING})


def build_fresh_tuning_pipeline(model_name: str, random_state: int) -> SkPipeline:
    """Novo pipeline (não treinado), mesma estrutura da etapa comparativa + GB com selector no tuning."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo não suportado para tuning: {model_name!r}")

    if model_name == DECISION_TREE:
        return SkPipeline([("model", DecisionTreeClassifier(random_state=random_state))])

    if model_name == RANDOM_FOREST:
        return SkPipeline([("model", RandomForestClassifier(random_state=random_state))])

    if model_name == SVM:
        return SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=random_state)),
            ]
        )

    return SkPipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k="all")),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]
    )


def param_distributions_for(model_name: str) -> dict[str, Any]:
    """Distribuições para `ParameterSampler` (listas = escolha uniforme entre opções)."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Modelo não suportado para tuning: {model_name!r}")

    if model_name == DECISION_TREE:
        return {
            "model__max_depth": randint(2, 30),
            "model__min_samples_split": randint(2, 80),
            "model__min_samples_leaf": randint(1, 40),
            "model__criterion": ["gini", "entropy"],
        }

    if model_name == RANDOM_FOREST:
        return {
            "model__n_estimators": randint(50, 400),
            "model__max_depth": randint(2, 30),
            "model__min_samples_split": randint(2, 40),
            "model__min_samples_leaf": randint(1, 30),
            "model__max_features": ["sqrt", "log2", None],
        }

    if model_name == SVM:
        return {
            "model__C": loguniform(1e-2, 1e3),
            "model__gamma": uniform(1e-4, 2.0),
        }

    # Gradient Boosting (grid já usado no projeto)
    return {
        "model__n_estimators": randint(50, 400),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__max_depth": randint(2, 6),
        "model__min_samples_split": randint(2, 100),
        "model__min_samples_leaf": randint(1, 50),
        "model__subsample": uniform(0.6, 0.4),
    }
