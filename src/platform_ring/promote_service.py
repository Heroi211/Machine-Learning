"""Promote generalizado por domínio (tabular FE + recomendação)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from ml_core_ring.domain_plugin import get_domain
from models.deployed_models import DeployedModels
from services.processor.deployment_service import promote_active_feature_engineering_for_objective


async def promote_for_domain(
    domain: str,
    promoted_by_user_id: int,
    db: AsyncSession,
    *,
    pipeline_run_id: int | None = None,
) -> DeployedModels:
    """
    Promove o modelo activo do domínio para servir em ``/predict``.

    - Tabular: run FE activo (comportamento Fase 01).
    - Recomendação: run ``pipeline_type=recommendation`` activo e concluído.
    """
    plugin = get_domain(domain)
    if plugin.problem_type == "binary_classification":
        if pipeline_run_id is not None:
            from services.processor.deployment_service import promote_pipeline_run

            return await promote_pipeline_run(
                domain=domain,
                pipeline_run_id=pipeline_run_id,
                promoted_by_user_id=promoted_by_user_id,
                pipeline_type="feature_engineering",
                db=db,
            )
        return await promote_active_feature_engineering_for_objective(
            objective=domain,
            promoted_by_user_id=promoted_by_user_id,
            db=db,
        )

    if plugin.problem_type == "recommendation":
        from services.processor.deployment_service import promote_recommendation_for_domain

        return await promote_recommendation_for_domain(
            domain=domain,
            promoted_by_user_id=promoted_by_user_id,
            db=db,
            pipeline_run_id=pipeline_run_id,
        )

    raise ValueError(f"problem_type {plugin.problem_type!r} sem promote definido.")
