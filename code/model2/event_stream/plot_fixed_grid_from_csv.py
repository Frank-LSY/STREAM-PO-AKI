#!/usr/bin/env python3
"""Plot fixed-grid bucket metric curves from an existing CSV (no model forward)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from event_stream.fixed_time_grid import plot_bucket_metric_curves


def load_csv_table(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = {}
            for k, v in r.items():
                if v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        if k in ("bucket_hour", "n", "n_pos", "n_neg"):
                            row[k] = int(float(v))
                        else:
                            row[k] = float(v)
                    except ValueError:
                        row[k] = v
            if "hour_mid" not in row or row["hour_mid"] is None:
                h0 = float(row.get("hour_start", 0))
                h1 = float(row.get("hour_end", h0 + 6))
                row["hour_mid"] = 0.5 * (h0 + h1)
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="fixed_time_grid_bucket_metrics.csv")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--bucket_size", type=float, default=6.0)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    table = load_csv_table(csv_path)
    paths = plot_bucket_metric_curves(table, out_dir, dpi=args.dpi, bucket_size=args.bucket_size)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
