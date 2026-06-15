"""Registo de domínios e backends de treino (import side-effect)."""

from domains.churn import plugin as churn_plugin  # noqa: F401
from domains.churn import train_backend as churn_train_backend  # noqa: F401
from domains.recommendation import plugin as recommendation_plugin  # noqa: F401
from domains.recommendation import train_backend as recommendation_train_backend  # noqa: F401

__all__ = [
    "churn_plugin",
    "churn_train_backend",
    "recommendation_plugin",
    "recommendation_train_backend",
]
