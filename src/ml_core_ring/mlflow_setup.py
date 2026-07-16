"""MLflow — tracking unificado (Postgres ``mlflow`` + artefactos partilhados)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Alinhado a ``mlflow_server --default-artifact-root`` e bind mount compose.
MLFLOW_ARTIFACT_CONTAINER_ROOT = "/mlflow/artifacts"

# Subpath no repo host (bind mount ./src/artifacts/mlruns → /mlflow/artifacts).
MLFLOW_ARTIFACT_HOST_SUBPATH = "src/artifacts/mlruns"


def uses_remote_mlflow_tracking(tracking_uri: str | None = None) -> bool:
    from core.configs import settings

    uri = (tracking_uri or settings.mlflow_tracking_uri or "").strip()
    return uri.startswith("http://") or uri.startswith("https://")


def resolved_mlflow_artifact_dir() -> str:
    """Directório local para ``mkdir`` / escrita directa (host ou contentor)."""
    from core.configs import settings

    raw = (settings.mlflow_artifact_root or MLFLOW_ARTIFACT_HOST_SUBPATH).strip()
    if raw in {MLFLOW_ARTIFACT_CONTAINER_ROOT, MLFLOW_ARTIFACT_CONTAINER_ROOT.rstrip("/")}:
        return MLFLOW_ARTIFACT_CONTAINER_ROOT
    if raw.startswith("/mlflow/"):
        return raw
    if os.path.isabs(raw):
        return raw
    from ml_core_ring.paths import ml_project_root

    return str(Path(ml_project_root()) / raw)


def configure_mlflow_tracking() -> None:
    import mlflow
    from core.configs import settings

    os.makedirs(resolved_mlflow_artifact_dir(), exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def ensure_mlflow_experiment(experiment_name: str) -> None:
    """Cria experimento sem ``artifact_location`` errado quando tracking é HTTP."""
    import mlflow

    configure_mlflow_tracking()
    if mlflow.get_experiment_by_name(experiment_name):
        mlflow.set_experiment(experiment_name)
        return
    if uses_remote_mlflow_tracking():
        # Servidor central: Postgres + --default-artifact-root /mlflow/artifacts
        mlflow.create_experiment(experiment_name)
    else:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=resolved_mlflow_artifact_dir(),
        )
    mlflow.set_experiment(experiment_name)


def log_artifact_resilient(local_path: str, artifact_path: str | None = None) -> None:
    """
    Regista artefacto via cliente MLflow; no host, faz fallback para ``src/artifacts/mlruns``.

    Com tracking HTTP, o experimento no servidor pode apontar ``file:///mlflow/artifacts/...``.
    Dentro do contentor isso funciona; no host o cliente falha em ``/mlflow`` — copiamos para o
    bind mount local (mesmo volume que o ``mlflow_server`` expõe).
    """
    import mlflow
    from ml_core_ring.paths import resolve_shared_artifact_path

    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        return
    except (PermissionError, OSError) as exc:
        run = mlflow.active_run()
        if run is None:
            raise
        uri = (run.info.artifact_uri or "").removeprefix("file://")
        if MLFLOW_ARTIFACT_CONTAINER_ROOT not in uri and "/mlflow/" not in str(exc):
            raise
        host_base = resolve_shared_artifact_path(uri)
        if not host_base:
            logger.warning("mlflow.log_artifact ignorado (sem mapeamento host): %s", exc)
            return
        dest_dir = Path(host_base)
        if artifact_path:
            dest_dir = dest_dir / artifact_path
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest_dir / Path(local_path).name)
        logger.info("Artefacto copiado para bind mount MLflow (host): %s", dest_dir)
