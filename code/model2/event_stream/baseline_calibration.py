"""Baseline ([BASE]) calibration curve helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from common.clinical_metrics import (
    _calibration_block,
    _draw_reliability_on_ax,
    apply_temperature_scaling,
    fit_temperature_scaling,
)
from event_stream.eval_common import build_eval_dataloader
from common.utils import collect_predictions


def collect_baseline_arrays(
    model: torch.nn.Module,
    episode_path: str,
    code_vocab: Dict[str, int],
    lab_stats: Any,
    num_med: int,
    batch_size: int,
    device: torch.device,
    age_stats: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = build_eval_dataloader(
        episode_path, code_vocab, lab_stats, num_med, batch_size, age_stats=age_stats
    )
    arrs = collect_predictions(model, loader, device)
    y = np.asarray(arrs["base_targets"], dtype=np.float64)
    p = np.asarray(arrs["base_probs"], dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    return y[m], p[m]


def fit_temperature_on_baseline(
    model: torch.nn.Module,
    calib_jsonl: str,
    code_vocab: Dict[str, int],
    lab_stats: Any,
    num_med: int,
    batch_size: int,
    device: torch.device,
    age_stats: Any = None,
) -> float:
    y, p = collect_baseline_arrays(
        model, calib_jsonl, code_vocab, lab_stats, num_med, batch_size, device, age_stats=age_stats
    )
    return fit_temperature_scaling(y, p)


def plot_baseline_calibration_curve(
    uncalibrated: Dict[str, Any],
    out_path: Path,
    *,
    temperature_scaled: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
    dpi: int = 150,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib required for calibration plots") from e

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
    _draw_reliability_on_ax(ax, uncalibrated, label="Baseline (uncalibrated)", linestyle="-", color="C0")
    if temperature_scaled is not None and abs(temperature - 1.0) > 1e-6:
        _draw_reliability_on_ax(
            ax,
            temperature_scaled,
            label=f"Baseline (T={temperature:.3f})",
            linestyle="--",
            color="C1",
        )
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, alpha=0.7, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted risk (mean probability per bin)")
    ax.set_ylabel("Observed risk (fraction AKI+)")
    n = uncalibrated.get("n", 0)
    ece = uncalibrated.get("ece")
    brier = uncalibrated.get("brier")
    slope = uncalibrated.get("calibration_slope")
    intercept = uncalibrated.get("calibration_intercept")
    subtitle = [
        f"n={n}",
        f"ECE={ece:.4f}" if ece is not None else "ECE=n/a",
        f"Brier={brier:.4f}" if brier is not None else "",
    ]
    if slope is not None and intercept is not None:
        subtitle.append(f"slope={slope:.3f}, intercept={intercept:.3f}")
    ax.set_title("Baseline AKI risk calibration\n" + " | ".join(s for s in subtitle if s))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_calibration_bins_csv(
    block: Dict[str, Any],
    path: Path,
    *,
    variant: str = "uncalibrated",
) -> None:
    curve = block.get("reliability_curve") or {}
    xs = curve.get("mean_predicted") or []
    ys = curve.get("fraction_positive") or []
    if not xs:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "bin_index", "mean_predicted_risk", "observed_risk", "n_bins_used"])
        n_used = curve.get("n_bins_used", len(xs))
        for i, (xp, yo) in enumerate(zip(xs, ys)):
            w.writerow([variant, i, xp, yo, n_used])


def run_baseline_calibration(
    model: torch.nn.Module,
    episode_path: str,
    out_dir: Path,
    code_vocab: Dict[str, int],
    lab_stats: Any,
    num_med: int,
    batch_size: int,
    device: torch.device,
    *,
    age_stats: Any = None,
    checkpoint: str,
    calib_jsonl: Optional[str] = None,
    n_bins: int = 10,
    plot_temperature_scaled: bool = True,
    export_csv: bool = False,
    dpi: int = 150,
) -> Dict[str, Any]:
    """Compute calibration metrics and optional plot/CSV under ``out_dir``."""
    from common.utils import save_json

    out_dir.mkdir(parents=True, exist_ok=True)
    temperature = 1.0
    temperature_fit_on: Optional[str] = None
    if calib_jsonl:
        temperature = fit_temperature_on_baseline(
            model, calib_jsonl, code_vocab, lab_stats, num_med, batch_size, device, age_stats=age_stats
        )
        temperature_fit_on = str(Path(calib_jsonl).resolve())
        print(f"Fitted temperature T={temperature:.4f} on {temperature_fit_on}")

    y, p = collect_baseline_arrays(
        model, episode_path, code_vocab, lab_stats, num_med, batch_size, device, age_stats=age_stats
    )
    uncal = _calibration_block(y, p, n_bins)
    p_cal = apply_temperature_scaling(p, temperature)
    scaled = _calibration_block(y, p_cal, n_bins) if abs(temperature - 1.0) > 1e-6 else None

    report: Dict[str, Any] = {
        "evaluation": "baseline_calibration",
        "prediction_unit": "one [BASE] token per episode",
        "checkpoint": str(Path(checkpoint).resolve()),
        "episode_path": str(Path(episode_path).resolve()),
        "n_bins": int(n_bins),
        "temperature_scaling_T": float(temperature),
        "temperature_fit_jsonl": temperature_fit_on,
        "uncalibrated": uncal,
        "temperature_scaled": scaled,
        "label_note": "Episode-level AKI label; x=mean predicted probability per bin, y=observed AKI+ rate.",
    }
    save_json(report, out_dir / "baseline_calibration.json")

    png_path = out_dir / "baseline_calibration_curve.png"
    plot_baseline_calibration_curve(
        uncal,
        png_path,
        temperature_scaled=scaled if plot_temperature_scaled else None,
        temperature=temperature,
        dpi=dpi,
    )
    print(f"Wrote {png_path}")

    if export_csv:
        write_calibration_bins_csv(uncal, out_dir / "baseline_calibration_bins.csv", variant="uncalibrated")
        if scaled is not None:
            write_calibration_bins_csv(
                scaled,
                out_dir / "baseline_calibration_bins_temperature_scaled.csv",
                variant="temperature_scaled",
            )

    print(
        f"Baseline calibration: n={uncal.get('n')} ECE={uncal.get('ece')} "
        f"Brier={uncal.get('brier')} slope={uncal.get('calibration_slope')}"
    )
    return report
