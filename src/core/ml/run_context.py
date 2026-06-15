"""Contexto e resultado de execução de pipelines modulares."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.ml.artifact_manifest import ArtifactManifest


@dataclass
class RunContext:
    domain: str
    params: dict[str, Any] = field(default_factory=dict)
    data_paths: dict[str, str] = field(default_factory=dict)
    mlflow_experiment: str = "default"
    random_state: int = 42
    stage: str | None = None


@dataclass
class ModelCandidate:
    name: str
    engine: str
    metrics: dict[str, float]
    manifest: ArtifactManifest
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    manifest: ArtifactManifest
    metrics: dict[str, float]
    mlflow_run_id: str | None = None
    champion_name: str | None = None
