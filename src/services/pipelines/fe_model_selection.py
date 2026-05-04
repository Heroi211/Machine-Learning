"""
Seleção do melhor estimador no FE com base numa métrica de otimização configurável.

Mantém a lógica de decisão fora do corpo de `train_models` / `tune` para facilitar testes e leitura.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ALLOWED_OPTIMIZATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1", "roc_auc"})

# Colunas produzidas em train_models (rótulos em PT para relatórios / MLflow)
_METRIC_TO_COLUMN: dict[str, str] = {
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC AUC",
}

_ALIASES: dict[str, str] = {
    "acuracia": "accuracy",
    "precisao": "precision",
}


def normalize_optimization_metric(metric: str) -> str:
    """Normaliza e valida a métrica enviada pela API (ex.: recall, accuracy)."""
    m = metric.strip().lower()
    m = _ALIASES.get(m, m)
    if m not in ALLOWED_OPTIMIZATION_METRICS:
        raise ValueError(
            f"Métrica de otimização inválida: {metric!r}. "
            f"Use uma de: {sorted(ALLOWED_OPTIMIZATION_METRICS)}"
        )
    return m


def result_column_for_metric(metric: str) -> str:
    """Nome da coluna no DataFrame de resultados (`Acurácia`, `Recall`, …)."""
    return _METRIC_TO_COLUMN[normalize_optimization_metric(metric)]


def sklearn_scoring_parameter(metric: str) -> str:
    """Parâmetro `scoring=` de `cross_val_score` / `permutation_importance` (nome sklearn)."""
    return normalize_optimization_metric(metric)


def test_set_score(
    y_true,
    y_pred,
    y_proba: np.ndarray | None,
    metric: str,
) -> float:
    """Pontuação no conjunto de teste alinhada à métrica de otimização."""
    m = normalize_optimization_metric(metric)
    if m == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if m == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if m == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if m == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if m == "roc_auc":
        if y_proba is None or y_proba.ndim < 2:
            return float("-inf")
        return float(roc_auc_score(y_true, y_proba[:, 1]))


def select_best_model_after_training(
    results_rows: list[dict[str, Any]],
    metric: str,
) -> tuple[str, float, dict[str, Any]]:
    """
    Escolhe o modelo com maior valor na métrica indicada.

    Parameters
    ----------
    results_rows
        Uma linha por modelo, com chaves 'Modelo' e colunas de métricas (ex.: 'Recall').
    metric
        Métrica de otimização normalizada (accuracy, recall, …).

    Returns
    -------
    model_name
        Nome amigável do modelo vencedor (mesma chave usada em `trained_models`).
    score
        Valor da métrica no teste para o vencedor.
    winner_row
        Dicionário completo da linha vencedora.
    """
    col = result_column_for_metric(metric)
    best_row: dict[str, Any] | None = None
    best_score = -np.inf

    for row in results_rows:
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        s = float(raw)
        if s > best_score:
            best_score = s
            best_row = row

    if best_row is None:
        raise ValueError(
            f"Nenhum modelo com valor válido para a coluna {col!r}; não é possível ranquear."
        )

    return best_row["Modelo"], best_score, best_row
