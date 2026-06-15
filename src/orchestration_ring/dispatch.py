"""Resolução de domínio e rota de treino para a DAG dispatch."""

from __future__ import annotations

from typing import Literal

from ml_core_ring.domain_plugin import get_domain

TrainingRoute = Literal["tabular", "recommendation"]


def resolve_domain(conf: dict) -> str:
    """Domínio efectivo: ``domain`` ou ``objective`` (legado tabular)."""
    raw = conf.get("domain") or conf.get("objective")
    if not raw:
        raise ValueError("Parâmetro 'domain' (ou 'objective' legado) é obrigatório no conf.")
    return str(raw).strip().lower()


def training_route_for_domain(domain: str) -> TrainingRoute:
    plugin = get_domain(domain)
    if plugin.problem_type == "recommendation":
        return "recommendation"
    return "tabular"


def validate_dispatch_conf(conf: dict) -> TrainingRoute:
    """Valida conf mínima e devolve rota tabular vs recomendação."""
    domain = resolve_domain(conf)
    route = training_route_for_domain(domain)

    if route == "tabular":
        csv_path = conf.get("csv_path")
        if not csv_path:
            raise ValueError(f"Domínio tabular {domain!r} exige 'csv_path' no conf.")
        if not __import__("os").path.isfile(str(csv_path)):
            raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    return route
