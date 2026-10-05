#!/usr/bin/env python3
"""Phase-2: re-score saved epoch histories under alternate selection weights.

Does not reload models. For each run's metrics_history.jsonl, recompute
  score(w) = w * macro_online + (1-w) * baseline_auprc
for w in {1.0, 0.9, 0.8}, pick the best epoch, and compare to the training-time
selection (w=0.9).

Also aggregates phase-1 training-λ grid (best under w=0.9) across folds.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from common.paths import PROJECT_ROOT


WEIGHTS = (1.0, 0.9, 0.8)


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # nan
        return None
    return v


def load_history(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def epoch_macro_base(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], int]:
    sel = row.get("selection") or {}
    macro = _f(sel.get("macro_online_hour_bucket_auprc"))
    base = _f(sel.get("baseline_auprc"))
    # fallbacks from val block
    val = row.get("val") or {}
    if base is None:
        base = _f((val.get("baseline_node") or {}).get("auprc"))
    epoch = int(row.get("epoch", -1))
    return macro, base, epoch


def best_under_weight(history: List[Dict[str, Any]], w: float) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    for row in history:
        macro, base, epoch = epoch_macro_base(row)
        if macro is None or base is None:
            continue
        if w >= 1.0 - 1e-12:
            score = macro
        elif w <= 1e-12:
            score = base
        else:
            score = w * macro + (1.0 - w) * base
        if score > best_score:
            best_score = score
            best = {
                "epoch": epoch,
                "score": score,
                "macro": macro,
                "baseline": base,
                "w_online": w,
            }
    return best


def parse_cfg_id(cfg_id: str) -> Tuple[float, float]:
    # lb0.3_lm0.01
    assert cfg_id.startswith("lb"), cfg_id
    rest = cfg_id[2:]
    lb_s, lm_s = rest.split("_lm", 1)
    return float(lb_s), float(lm_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=(PROJECT_ROOT / "result/event_stream/sensitivity/lambda_grid"),
    )
    ap.add_argument("--folds", type=str, default="1 2 3 4 5")
    ap.add_argument(
        "--out",
        type=Path,
        default=(PROJECT_ROOT / "result/event_stream/sensitivity"),
    )
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.replace(",", " ").split()]
    args.out.mkdir(parents=True, exist_ok=True)

    # --- Phase 1 table: training λ at fixed w=0.9 ---
    phase1_rows: List[Dict[str, Any]] = []
    # --- Phase 2: selection weight sensitivity ---
    phase2_rows: List[Dict[str, Any]] = []

    cfg_dirs = sorted([p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("lb")])
    for cfg_dir in cfg_dirs:
        cfg_id = cfg_dir.name
        try:
            lb, lm = parse_cfg_id(cfg_id)
        except Exception:
            continue
        fold_macros, fold_bases, fold_scores, fold_epochs = [], [], [], []
        for fold in folds:
            hist_path = cfg_dir / f"fold{fold}" / "metrics_history.jsonl"
            history = load_history(hist_path)
            if not history:
                continue
            # training-time selection used λ=0.9
            b09 = best_under_weight(history, 0.9)
            if b09 is None:
                continue
            fold_macros.append(b09["macro"])
            fold_bases.append(b09["baseline"])
            fold_scores.append(b09["score"])
            fold_epochs.append(b09["epoch"])

            train_epoch = b09["epoch"]
            for w in WEIGHTS:
                bw = best_under_weight(history, w)
                if bw is None:
                    continue
                phase2_rows.append(
                    {
                        "cfg_id": cfg_id,
                        "lambda_base": lb,
                        "lambda_mem": lm,
                        "fold": fold,
                        "w_online": w,
                        "selected_epoch": bw["epoch"],
                        "score": bw["score"],
                        "macro": bw["macro"],
                        "baseline": bw["baseline"],
                        "same_epoch_as_w0.9": int(bw["epoch"] == train_epoch),
                        "epoch_delta_vs_w0.9": int(bw["epoch"] - train_epoch),
                    }
                )

        if fold_scores:
            phase1_rows.append(
                {
                    "cfg_id": cfg_id,
                    "lambda_base": lb,
                    "lambda_mem": lm,
                    "n_folds": len(fold_scores),
                    "sel_score_mean": st.mean(fold_scores),
                    "sel_score_std": st.stdev(fold_scores) if len(fold_scores) > 1 else 0.0,
                    "macro_mean": st.mean(fold_macros),
                    "macro_std": st.stdev(fold_macros) if len(fold_macros) > 1 else 0.0,
                    "baseline_mean": st.mean(fold_bases),
                    "baseline_std": st.stdev(fold_bases) if len(fold_bases) > 1 else 0.0,
                    "mean_best_epoch": st.mean(fold_epochs),
                }
            )

    phase1_rows.sort(key=lambda r: (-r["sel_score_mean"], r["lambda_base"], r["lambda_mem"]))

    p1_csv = args.out / "phase1_training_lambda_grid.csv"
    if phase1_rows:
        with p1_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(phase1_rows[0].keys()))
            w.writeheader()
            w.writerows(phase1_rows)

    p2_csv = args.out / "phase2_selection_weight_rescore.csv"
    if phase2_rows:
        with p2_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(phase2_rows[0].keys()))
            w.writeheader()
            w.writerows(phase2_rows)

    # Summary: fraction of runs where w changes selected epoch
    summary: Dict[str, Any] = {
        "phase1_n_configs": len(phase1_rows),
        "phase2_n_rows": len(phase2_rows),
        "phase1_csv": str(p1_csv),
        "phase2_csv": str(p2_csv),
        "top_configs_by_sel_score": phase1_rows[:5],
        "selection_agreement": {},
    }
    for w in WEIGHTS:
        subset = [r for r in phase2_rows if abs(r["w_online"] - w) < 1e-9]
        if not subset:
            continue
        same = sum(r["same_epoch_as_w0.9"] for r in subset)
        summary["selection_agreement"][str(w)] = {
            "n": len(subset),
            "frac_same_epoch_as_0.9": same / len(subset),
            "mean_abs_epoch_delta": st.mean([abs(r["epoch_delta_vs_w0.9"]) for r in subset]),
        }

    # Stable region heuristic: configs within 0.01 of best mean sel score
    if phase1_rows:
        best = phase1_rows[0]["sel_score_mean"]
        stable = [
            r
            for r in phase1_rows
            if r["sel_score_mean"] >= best - 0.01
        ]
        summary["stable_region_within_0.01_of_best"] = [
            {
                "cfg_id": r["cfg_id"],
                "lambda_base": r["lambda_base"],
                "lambda_mem": r["lambda_mem"],
                "sel_score_mean": r["sel_score_mean"],
                "macro_mean": r["macro_mean"],
                "baseline_mean": r["baseline_mean"],
            }
            for r in stable
        ]

    (args.out / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    lines = [
        "# Training-λ / selection-weight sensitivity",
        "",
        "## Protocol",
        "",
        "1. **Phase 1 (retrain):** fix selection \(w_{online}=0.9\), sweep "
        "\(λ_{base}\\in\{0.1,0.3,0.5\}\), \(λ_{MEM}\\in\{0,0.005,0.01,0.02\}\).",
        "2. **Phase 2 (no retrain):** for each run, re-pick best epoch under "
        "\(w\\in\{1.0,0.9,0.8\}\) using saved `metrics_history.jsonl`.",
        "",
        f"Folds: {folds}. Embedding: production defaults (`num_med_exposure=0`, age MLP, hour-only time).",
        "",
        "## Phase 1 — training objective (selected under w=0.9)",
        "",
        "| cfg | λ_base | λ_mem | n | sel↓ | macro | baseline | mean epoch |",
        "|-----|-------:|------:|--:|-----:|------:|---------:|-----------:|",
    ]
    for r in phase1_rows:
        lines.append(
            f"| {r['cfg_id']} | {r['lambda_base']} | {r['lambda_mem']} | {r['n_folds']} | "
            f"{r['sel_score_mean']:.4f}±{r['sel_score_std']:.3f} | "
            f"{r['macro_mean']:.4f} | {r['baseline_mean']:.4f} | {r['mean_best_epoch']:.1f} |"
        )
    lines += ["", "## Phase 2 — selection-weight agreement vs w=0.9", ""]
    for w, info in summary.get("selection_agreement", {}).items():
        lines.append(
            f"- **w={w}**: {info['frac_same_epoch_as_0.9']*100:.1f}% same epoch "
            f"(n={info['n']}), mean |Δepoch|={info['mean_abs_epoch_delta']:.2f}"
        )
    if summary.get("stable_region_within_0.01_of_best"):
        lines += ["", "## Stable region (|Δsel| ≤ 0.01 from best)", ""]
        for r in summary["stable_region_within_0.01_of_best"]:
            lines.append(
                f"- `{r['cfg_id']}`: sel={r['sel_score_mean']:.4f}, "
                f"macro={r['macro_mean']:.4f}, base={r['baseline_mean']:.4f}"
            )
    lines += [
        "",
        "## Interpretation template",
        "",
        "If Phase 1 shows a wide plateau over \(λ_{base}\\in[0.1,0.5]\) and nearby MEM, "
        "and Phase 2 keeps similar epochs for \(w\\in\{0.8,0.9,1.0\}\), conclude:",
        "",
        "> Training and checkpoint-selection weights were chosen to prioritize the "
        "postoperative online task while retaining baseline prediction as a secondary "
        "objective, and performance was stable across nearby weighting choices.",
        "",
        f"Artifacts: `{p1_csv.name}`, `{p2_csv.name}`, `sensitivity_summary.json`.",
        "",
    ]
    (args.out / "SENSITIVITY_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2)[:2000])
    print(f"Wrote {p1_csv}")
    print(f"Wrote {p2_csv}")
    print(f"Wrote {args.out / 'SENSITIVITY_REPORT.md'}")


if __name__ == "__main__":
    main()
