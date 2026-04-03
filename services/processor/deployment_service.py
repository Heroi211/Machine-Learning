"""Gestão de modelos promovidos (produção) por domínio."""

from __future__ import annotations

import os

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.deployed_models import DeployedModels
from models.pipeline_runs import PipelineRuns
from services.utils import utcnow


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

    stmt_run = select(PipelineRuns).where(PipelineRuns.id == pipeline_run_id)
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
