"""Registo de domínios (import side-effect)."""

from domains.churn import plugin as churn_plugin  # noqa: F401
from domains.recommendation import plugin as recommendation_plugin  # noqa: F401

__all__ = ["churn_plugin", "recommendation_plugin"]
