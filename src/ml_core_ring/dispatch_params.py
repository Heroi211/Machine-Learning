"""Normalização de conf Airflow/API → parâmetros de treino do executor."""

from __future__ import annotations

from typing import Any

# Chaves de routing / plataforma — não são hiperparâmetros ML.
_DISPATCH_META_KEYS = frozenset(
    {
        "domain",
        "objective",
        "user_id",
        "csv_path",
        "auto_promote",
        "optimization_metric",
        "time_limit_minutes",
        "tuning_n_iter",
        "acc_target",
        "min_precision",
        "min_roc_auc",
        "decision_threshold",
    }
)


def flatten_dispatch_train_params(conf: dict[str, Any]) -> dict[str, Any]:
    """
    Extrai parâmetros de treino a partir do conf do dispatch.

    Suporta conf plana (``train_models`` no topo) e aninhada (``params: {...}``).
    Valores no topo do conf sobrescrevem chaves dentro de ``params``.
    """
    merged: dict[str, Any] = {}
    nested = conf.get("params")
    if isinstance(nested, dict):
        merged.update(nested)

    for key, value in conf.items():
        if key in _DISPATCH_META_KEYS or key == "params":
            continue
        merged[key] = value

    return merged
