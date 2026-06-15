"""Bootstrap de paths para workers Airflow (partilhado entre DAGs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ML_PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
ML_CODE_ROOT = os.environ.get("ML_CODE_ROOT", "").strip()
DEFAULT_PIPELINE_USER_ID = int(os.environ.get("ML_PIPELINE_DEFAULT_USER_ID", "2"))


def bootstrap_ml_sys_path() -> None:
    """Garante ``ML_PROJECT_ROOT`` e ``src/`` no ``sys.path``."""
    code_root = ML_CODE_ROOT
    if not code_root:
        candidate = Path(__file__).resolve().parents[2] / "src"
        if candidate.is_dir():
            code_root = str(candidate)
    for path in (ML_PROJECT_ROOT, code_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def prepend_airflow_ml_site_packages() -> None:
    """Bibliotecas extra do projecto (``/opt/airflow/ml_libs``)."""
    root = os.environ.get("ML_AIRFLOW_SITE_PACKAGES", "/opt/airflow/ml_libs").strip()
    if root and os.path.isdir(root):
        resolved = os.path.abspath(root)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def resolve_ml_artifact_path(path: str) -> str:
    """Resolve caminhos relativos ao ``ML_PROJECT_ROOT``."""
    path = os.path.normpath(path.strip())
    if os.path.isfile(path):
        return os.path.abspath(path)
    under_root = os.path.join(ML_PROJECT_ROOT, path)
    if os.path.isfile(under_root):
        return os.path.abspath(under_root)
    abs_cwd = os.path.abspath(path)
    if os.path.isfile(abs_cwd):
        return abs_cwd
    return os.path.abspath(under_root)
