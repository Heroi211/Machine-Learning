"""Resolução de caminhos de artefactos partilhados entre contentores e API local."""

from __future__ import annotations

import os
from pathlib import Path

# Prefixos gravados na BD pelo Airflow / worker (volume ml_shared ≡ ml_project no compose).
_AIRFLOW_ML_PREFIX = "/opt/airflow/ml_project/"
_API_ML_SHARED_PREFIX = "/var/www/ml_shared/"
_MLFLOW_ARTIFACT_PREFIX = "/mlflow/artifacts"


def ml_project_root() -> str:
    from core.configs import settings

    return (settings.ml_project_root or ".").rstrip("/")


def resolved_ml_shared_uploads_dir() -> str:
    """Diretório de upload no host (``ml_data/uploads`` por omissão)."""
    from core.configs import settings

    raw = (settings.ml_shared_path or "ml_data/uploads").strip()
    if os.path.isabs(raw):
        return raw
    return str(Path(ml_project_root()) / raw)


def airflow_upload_path(filename: str) -> str:
    """Caminho do CSV no conf Airflow (volume partilhado com o host)."""
    return f"{_AIRFLOW_ML_PREFIX}uploads/{filename}"


def resolve_shared_artifact_path(path: str | None) -> str | None:
    """
    Traduz prefixos cross-container para ``ML_PROJECT_ROOT`` actual.

    Funciona na API Docker (``/var/www/ml_shared``) e em uvicorn local (repo root).
    """
    if not path:
        return path
    root = ml_project_root().rstrip("/") + "/"
    if path.startswith(_MLFLOW_ARTIFACT_PREFIX):
        suffix = path[len(_MLFLOW_ARTIFACT_PREFIX) :].lstrip("/")
        base = f"{root}src/artifacts/mlruns"
        return f"{base}/{suffix}" if suffix else base
    for src in (_AIRFLOW_ML_PREFIX, _API_ML_SHARED_PREFIX):
        if path.startswith(src):
            return root + path[len(src) :]
    return path
