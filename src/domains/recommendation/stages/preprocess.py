"""Entrypoint DVC — preprocess."""

from __future__ import annotations

import logging

from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ctx = build_run_context()
    RecommendationPipelineRunner().run_stage("preprocess", ctx)


if __name__ == "__main__":
    main()
