"""
Drift de dados: compara CSV de treino (referência) com produção (predições).

Produção: CSV exportado da tabela `predictions` com coluna `input_data` (JSON)
por linha, ou CSV já com colunas de features alinhadas ao treino.
Grava PSI e resumo em PATH_MAINTENANCE_REPORTS.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configs import settings  # noqa: E402


def _calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    expected = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    actual = np.histogram(current, bins=breakpoints)[0] / len(current)
    expected = np.where(expected == 0, 0.0001, expected)
    actual = np.where(actual == 0, 0.0001, actual)
    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(psi)


def _load_predictions_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input_data" in df.columns:
        parsed = df["input_data"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return pd.json_normalize(parsed)
    drop_cols = [c for c in ("id", "prediction", "probability", "pipeline_run_id") if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift treino vs predições (CSV)")
    parser.add_argument("--train-csv", required=True, type=Path, help="CSV de treino (referência)")
    parser.add_argument("--predictions-csv", required=True, type=Path, help="Export de predictions ou features")
    parser.add_argument("--target-col", default="target", help="Coluna alvo no treino (removida na comparação)")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    if "dataset" in train.columns:
        train = train.drop(columns=["dataset"])
    if args.target_col in train.columns:
        train = train.drop(columns=[args.target_col])

    prod = _load_predictions_features(args.predictions_csv)

    numeric_cols = [c for c in train.columns if c in prod.columns and pd.api.types.is_numeric_dtype(train[c])]
    if not numeric_cols:
        print("Nenhuma coluna numérica em comum entre treino e produção.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for col in numeric_cols:
        ref = train[col].dropna().astype(float).values
        cur = prod[col].dropna().astype(float).values
        if len(ref) < 2 or len(cur) < 2:
            continue
        psi = _calculate_psi(ref, cur)
        rows.append({"feature": col, "psi": psi})

    psi_df = pd.DataFrame(rows)
    out_dir = Path(settings.path_maintenance_reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"drift_psi_{stamp}.csv"
    psi_df.to_csv(out_csv, index=False)
    print(f"PSI: {out_csv}")
    print(psi_df.to_string(index=False))


if __name__ == "__main__":
    main()
