"""Routers HTTP por domínio (Fase 5.5)."""

from __future__ import annotations

from fastapi import APIRouter

from platform_ring.domains.churn.router import router as churn_router
from platform_ring.domains.recommendation.router import router as recommendation_router

domains_router = APIRouter()
domains_router.include_router(churn_router)
domains_router.include_router(recommendation_router)

__all__ = ["domains_router"]
