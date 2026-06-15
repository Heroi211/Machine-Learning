#!/usr/bin/env python3
"""Promove o modelo TC02 no MLflow Model Registry (Staging → Production)."""

from __future__ import annotations

import argparse
import os

from mlflow import MlflowClient
from mlflow.tracking import set_tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Promove tc02_recommender para Production.")
    parser.add_argument("--model-name", default="tc02_recommender")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///src/artifacts/mlruns/mlflow.db"),
    )
    args = parser.parse_args()

    set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{args.model_name}'")
    if not versions:
        raise SystemExit(f"Nenhuma versão registrada para {args.model_name!r}. Execute dvc repro + train antes.")

    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version(
        name=args.model_name,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version(
        name=args.model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Modelo {args.model_name} v{latest.version} promovido para Production.")


if __name__ == "__main__":
    main()
