"""Entrypoint DVC — feature_eng."""

from __future__ import annotations

import logging
from pathlib import Path

from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ctx = build_run_context()
    runner = RecommendationPipelineRunner()
    interactions = Path("data/recommendation/processed/interactions.parquet")
    if not interactions.is_file():
        interactions = runner.preprocess(ctx)
    runner.feature_eng(interactions, ctx)


if __name__ == "__main__":
    main()
