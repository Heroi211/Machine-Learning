"""Gestão de modelos promovidos (produção) por domínio."""

from __future__ import annotations

import os

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.configs import settings
from models.deployed_models import DeployedModels
from models.pipeline_runs import PipelineRuns
from services.utils import utcnow


class RollbackError(Exception):
    """Não existe deployment archived para restaurar no domínio."""


class NoActiveDeploymentError(Exception):
    """Não existe modelo ativo para o domínio solicitado."""


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower()


async def get_active_deployment(domain: str, db: AsyncSession) -> DeployedModels | None:
    d = _normalize_domain(domain)
    stmt = (
        select(DeployedModels)
        .where(
            DeployedModels.domain == d,
            DeployedModels.status == "active",
            DeployedModels.active.is_(True),
        )
        .options(selectinload(DeployedModels.pipeline_run))
    )
    result = await db.execute(stmt)
    return result.scalars().one_or_none()


async def promote_active_feature_engineering_for_objective(
    objective: str,
    promoted_by_user_id: int,
    db: AsyncSession,
) -> DeployedModels:
    """
    Promove o único ``PipelineRuns`` de feature engineering **activo**, concluído e com o ``objective``
    dado (tipicamente ``settings.objective``). Exige exactamente um candidato.
    """
    obj = _normalize_domain(objective)
    predicates = [
        PipelineRuns.pipeline_type == "feature_engineering",
        PipelineRuns.status == "completed",
        PipelineRuns.active.is_(True),
        func.lower(PipelineRuns.objective) == obj,
    ]
    if settings.is_production:
        predicates.append(PipelineRuns.is_airflow_run.is_(True))
    stmt = select(PipelineRuns).where(and_(*predicates))
    res = await db.execute(stmt)
    runs = list(res.scalars().all())
    if len(runs) == 0:
        raise ValueError(
            "Nenhum pipeline feature_engineering activo e concluído para este objective. "
            "Execute o FE até haver um run vencedor (cv_*) ou reveja o estado em pipeline_runs."
        )
    if len(runs) > 1:
        raise ValueError(
            f"Ambiguidade: {len(runs)} runs FE activos para objective={objective!r}. "
            "É necessário exactamente um; desactive os obsoletos antes de promover."
        )
    run_id = runs[0].id
    return await promote_pipeline_run(
        domain=objective,
        pipeline_run_id=run_id,
        promoted_by_user_id=promoted_by_user_id,
        pipeline_type="feature_engineering",
        db=db,
    )


async def promote_pipeline_run(
    domain: str,
    pipeline_run_id: int,
    promoted_by_user_id: int,
    pipeline_type: str,
    db: AsyncSession,
) -> DeployedModels:
    """
    Arquiva o deployment ativo anterior do domínio e cria um novo como active.
    ``pipeline_type`` deve ser ``feature_engineering`` e o run na BD tem de coincidir.
    """
    d = _normalize_domain(domain)
    pt = pipeline_type.strip().lower()
    if pt != "feature_engineering":
        raise ValueError("Apenas pipeline_type='feature_engineering' pode ser promovido.")

    stmt_run = select(PipelineRuns).where(PipelineRuns.id == pipeline_run_id, PipelineRuns.active==True)
    res = await db.execute(stmt_run)
    run = res.scalars().one_or_none()
    if not run:
        raise ValueError(f"PipelineRun {pipeline_run_id} não encontrado.")
    if run.status != "completed":
        raise ValueError(f"PipelineRun {pipeline_run_id} não está concluído (status={run.status}).")
    if run.pipeline_type != pt:
        raise ValueError(
            f"PipelineRun {pipeline_run_id} é do tipo '{run.pipeline_type}', "
            f"mas foi pedida promoção como '{pt}'."
        )
    if _normalize_domain(run.objective) != d:
        raise ValueError(
            f"Domínio '{domain}' não coincide com o objective do run ('{run.objective}')."
        )
    if not run.model_path or not os.path.exists(run.model_path):
        raise ValueError(f"Artefato do modelo não encontrado em: {run.model_path}")

    await db.execute(
        update(DeployedModels)
        .where(
            DeployedModels.domain == d,
            DeployedModels.status == "active",
        )
        .values(status="archived")
    )

    row = DeployedModels(
        domain=d,
        pipeline_run_id=pipeline_run_id,
        status="active",
        promoted_at=utcnow(),
        promoted_by_user_id=promoted_by_user_id,
        metrics_snapshot=run.metrics,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    await db.refresh(row, attribute_names=["pipeline_run"])
    return row


async def get_deployment_history(domain: str, db: AsyncSession, limit: int = 10) -> list[DeployedModels]:
    """Retorna os últimos `limit` deployments (active + archived) para o domínio, do mais recente ao mais antigo."""
    d = _normalize_domain(domain)
    stmt = (
        select(DeployedModels)
        .where(DeployedModels.domain == d)
        .order_by(DeployedModels.promoted_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def rollback_deployment(domain: str, db: AsyncSession) -> DeployedModels:
    """
    Reverte para o deployment archived mais recente do domínio.
    Arquiva o active atual e reativa o último archived.
    """
    d = _normalize_domain(domain)

    stmt_archived = (
        select(DeployedModels)
        .where(DeployedModels.domain == d, DeployedModels.status == "archived")
        .order_by(DeployedModels.promoted_at.desc())
        .limit(1)
    )
    res = await db.execute(stmt_archived)
    previous = res.scalars().one_or_none()
    if not previous:
        raise RollbackError(f"Nenhum deployment archived encontrado para o domínio '{domain}'.")

    await db.execute(
        update(DeployedModels)
        .where(DeployedModels.domain == d, DeployedModels.status == "active")
        .values(status="archived")
    )

    previous.status = "active"
    db.add(previous)
    await db.commit()
    await db.refresh(previous)
    return previous
