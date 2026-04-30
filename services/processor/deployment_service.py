"""Gestão de modelos promovidos (produção) por domínio."""

from __future__ import annotations

import os

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.deployed_models import DeployedModels
from models.pipeline_runs import PipelineRuns
from services.utils import utcnow


class RollbackError(Exception):
    """Não existe deployment archived para restaurar no domínio."""


class NoActiveDeploymentError(Exception):
    """Não existe modelo ativo para o domínio solicitado."""


def _normalize_domain(domain: str) -> str:
    """Normalize domain names before persistence or lookup."""
    return domain.strip().lower()


async def get_active_deployment(domain: str, db: AsyncSession) -> DeployedModels | None:
    """Return the active deployment for a domain, if it exists."""
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


async def promote_pipeline_run(
    domain: str,
    pipeline_run_id: int,
    promoted_by_user_id: int,
    db: AsyncSession,
) -> DeployedModels:
    """
    Arquiva o deployment ativo anterior do domínio e cria um novo como active.
    Exige que o run exista, esteja completed e que objective case-insensitive == domain.
    """
    d = _normalize_domain(domain)

    stmt_run = select(PipelineRuns).where(PipelineRuns.id == pipeline_run_id, PipelineRuns.active==True)
    res = await db.execute(stmt_run)
    run = res.scalars().one_or_none()
    if not run:
        raise ValueError(f"PipelineRun {pipeline_run_id} não encontrado.")
    if run.status != "completed":
        raise ValueError(f"PipelineRun {pipeline_run_id} não está concluído (status={run.status}).")
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
