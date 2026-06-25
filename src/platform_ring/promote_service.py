"""Promote generalizado por domínio (tabular FE + recomendação)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from ml_core_ring.domain_plugin import get_domain
from models.deployed_models import DeployedModels
from platform_ring.mlflow_registry import RegistryPromoteResult, sync_mlflow_registry_on_promote
from services.processor.deployment_service import promote_active_feature_engineering_for_objective


@dataclass(frozen=True)
class PromoteForDomainResult:
    deployment: DeployedModels
    mlflow_registry: RegistryPromoteResult | None = None


async def promote_for_domain(
    domain: str,
    promoted_by_user_id: int,
    db: AsyncSession,
    *,
    pipeline_run_id: int | None = None,
) -> PromoteForDomainResult:
    """
    Promove o modelo activo do domínio para servir em ``/predict``.

    - Tabular: run FE activo (comportamento Fase 01).
    - Recomendação: run ``pipeline_type=recommendation`` activo e concluído.
    """
    plugin = get_domain(domain)
    if plugin.problem_type == "binary_classification":
        if pipeline_run_id is not None:
            from services.processor.deployment_service import promote_pipeline_run

            deployment = await promote_pipeline_run(
                domain=domain,
                pipeline_run_id=pipeline_run_id,
                promoted_by_user_id=promoted_by_user_id,
                pipeline_type="feature_engineering",
                db=db,
            )
        else:
            deployment = await promote_active_feature_engineering_for_objective(
                objective=domain,
                promoted_by_user_id=promoted_by_user_id,
                db=db,
            )
        registry = sync_mlflow_registry_on_promote(
            domain=domain,
            metrics=deployment.metrics_snapshot,
            pipeline_type="feature_engineering",
        )
        return PromoteForDomainResult(deployment=deployment, mlflow_registry=registry)

    if plugin.problem_type == "recommendation":
        from services.processor.deployment_service import promote_recommendation_for_domain

        deployment = await promote_recommendation_for_domain(
            domain=domain,
            promoted_by_user_id=promoted_by_user_id,
            db=db,
            pipeline_run_id=pipeline_run_id,
        )
        registry = sync_mlflow_registry_on_promote(
            domain=domain,
            metrics=deployment.metrics_snapshot,
            pipeline_type="recommendation",
        )
        return PromoteForDomainResult(deployment=deployment, mlflow_registry=registry)

    raise ValueError(f"problem_type {plugin.problem_type!r} sem promote definido.")
