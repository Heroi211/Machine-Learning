"""Testes de registro de domínios."""

from domains import churn_plugin, recommendation_plugin  # noqa: F401
from core.ml.domain_plugin import DOMAIN_REGISTRY, get_domain


def test_domain_registry_has_churn_and_recommendation():
    assert "churn" in DOMAIN_REGISTRY
    assert "recommendation" in DOMAIN_REGISTRY
    assert get_domain("churn").problem_type == "binary_classification"
    assert get_domain("recommendation").problem_type == "recommendation"
