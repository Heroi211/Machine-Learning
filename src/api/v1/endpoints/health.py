"""Endpoint público de health check da API e do banco de dados."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.deps import get_session

router = APIRouter()


def _meta() -> dict[str, str]:
    """Monta metadados básicos do serviço expostos no health check."""
    return {
        "service": settings.project_name,
        "version": settings.project_version.strip() or settings.project_version,
        "environment": settings.environment,
    }


@router.get(
    "",
    summary="Health",
    description=(
        "Sem autenticação. **200** só após `SELECT 1` no Postgres — processo vivo **e** banco acessível; "
        "**503** se o banco falhar. Rota única intencional para o escopo acadêmico "
        "(sem separar liveness/readiness)."
    ),
    status_code=status.HTTP_200_OK,
)
async def health(db: AsyncSession = Depends(get_session)) -> dict:
    """Confirma que a API está ativa e que o Postgres responde."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Banco de dados indisponível.",
        )
    return {"alive": True, "database": "up", **_meta()}
