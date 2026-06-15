"""Resolução de caminhos de artefactos partilhados entre contentores."""

from __future__ import annotations

SHARED_PATH_REMAP: tuple[tuple[str, str], ...] = (
    ("/opt/airflow/ml_project/", "/var/www/ml_shared/"),
)


def resolve_shared_artifact_path(path: str | None) -> str | None:
    """Traduz prefixos cross-container do volume partilhado ml_shared."""
    if not path:
        return path
    for src, dst in SHARED_PATH_REMAP:
        if path.startswith(src):
            return dst + path[len(src) :]
    return path
