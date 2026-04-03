"""
Agrega latência a partir de access.jsonl (e rotacionados access.jsonl.*).
Grava resumo CSV em PATH_MAINTENANCE_REPORTS.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configs import settings  # noqa: E402


def _load_jsonl_files(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_dir.is_dir():
        return rows
    for name in sorted(log_dir.glob("access.jsonl*")):
        if not name.is_file():
            continue
        with open(name, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _normalize_latency_rows(raw: list[dict]) -> pd.DataFrame:
    """Mapeia campos do middleware atual para colunas estáveis de análise."""
    out = []
    for r in raw:
        if "duration_ms" not in r:
            continue
        out.append(
            {
                "timestamp": r.get("ts") or r.get("timestamp"),
                "latency_ms": float(r["duration_ms"]),
                "status_code": r.get("status") if r.get("status") is not None else r.get("status_code"),
                "request_id": r.get("request_id"),
                "path": r.get("path"),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    log_dir = Path(settings.path_api_request_logs)
    raw = _load_jsonl_files(log_dir)
    if not raw:
        print(f"Nenhuma linha válida em {log_dir} (access.jsonl*)", file=sys.stderr)
        sys.exit(1)

    df = _normalize_latency_rows(raw)
    if df.empty:
        print("Nenhuma coluna duration_ms encontrada.", file=sys.stderr)
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    q = df["latency_ms"].quantile([0.5, 0.9, 0.95, 0.99])
    summary = pd.DataFrame(
        {
            "metric": ["count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"],
            "value": [
                len(df),
                float(df["latency_ms"].mean()),
                float(q.loc[0.5]),
                float(q.loc[0.9]),
                float(q.loc[0.95]),
                float(q.loc[0.99]),
            ],
        }
    )

    out_dir = Path(settings.path_maintenance_reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"latency_summary_{stamp}.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Resumo: {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
