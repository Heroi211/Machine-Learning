"""Entrypoint DVC — evaluate (+ MLflow)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ctx = build_run_context()
    runner = RecommendationPipelineRunner()
    metrics_path = Path("models/recommendation/train_metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError("Execute train antes de evaluate.")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    from core.ml.artifact_manifest import ArtifactManifest
    from core.ml.run_context import ModelCandidate

    candidates = [
        ModelCandidate(
            name=name,
            engine=name.split("_")[0] if "_" in name else name,
            metrics=metrics,
            manifest=ArtifactManifest(
                domain="recommendation",
                problem_type="recommendation",
                engine="torch_embedding" if "torch" in name else f"sklearn_{name}",
                artifacts={},
                metadata={"backend": name},
            ),
        )
        for name, metrics in raw.items()
    ]
    champion = runner.evaluate(candidates, ctx)
    runner.log_mlflow(champion, ctx)


if __name__ == "__main__":
    main()
