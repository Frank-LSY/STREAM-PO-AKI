#!/usr/bin/env python3
"""Generate one AKI+ and one AKI− clinical case-study figure each.

Panels: online risk with threshold crossings / T_alert / T_AKI annotated on the x-axis
(at the corresponding postop event), summary text, optional occlusion attribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.clinical_metrics import EpisodeTrajectory, first_alert_time, trajectory_from_episode_and_forward
from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_eval_dataloader, build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_attribution import attributions_for_episode
from event_stream.input_builder import episode_to_sequence_features
from common.med_exposure import infer_num_med_exposure
from common.utils import _forward_model, resolve_torch_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AKI+ and AKI− clinical case studies.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--alert_tau", type=float, default=0.5)
    p.add_argument("--person_id_pos", type=str, default=None, help="Force AKI+ case person_id.")
    p.add_argument("--person_id_neg", type=str, default=None, help="Force AKI− case person_id.")
    p.add_argument(
        "--with_attribution",
        action="store_true",
        help="Run occlusion attribution for explanation panel (slow).",
    )
    p.add_argument("--max_preop_occlusions", type=int, default=40)
    p.add_argument("--max_postop_query_steps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all episodes (for case selection).")
    return p.parse_args()


def collect_trajectories(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    episodes: List[Dict[str, Any]],
) -> Tuple[List[EpisodeTrajectory], Dict[str, Dict[str, Any]]]:
    model.eval()
    trajectories: List[EpisodeTrajectory] = []
    ep_by_id: Dict[str, Dict[str, Any]] = {}
    ep_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = _forward_model(model, batch)
            token_probs = torch.sigmoid(outputs["token_logits"]).cpu().numpy()
            base_logits = outputs["baseline_logits_all"]
            B = token_probs.shape[0]
            for i in range(B):
                if ep_idx >= len(episodes):
                    break
                episode = episodes[ep_idx]
                pid = str(episode.get("person_id", ep_idx))
                ep_by_id[pid] = episode
                feat = episode_to_sequence_features(
                    episode,
                    loader.dataset.code_vocab,  # type: ignore[attr-defined]
                    add_base=True,
                    lab_stats=getattr(loader.dataset, "lab_stats", None),  # type: ignore
                    num_med_exposure=getattr(loader.dataset, "num_med_exposure", 0),  # type: ignore
                )
                seq = feat["sequence"]
                valid_len = int(batch["attention_mask"][i].sum().item())
                probs_i = token_probs[i, :valid_len]
                base_mask = batch["base_mask"][i, :valid_len].cpu().numpy() > 0
                if base_mask.any():
                    bp = float(torch.sigmoid(base_logits[i, :valid_len][base_mask]).max().item())
                else:
                    bp = float(torch.sigmoid(base_logits[i, :valid_len]).max().item())
                trajectories.append(trajectory_from_episode_and_forward(episode, probs_i, bp, seq))
                ep_idx += 1
    return trajectories, ep_by_id


def _lead_time_hours(tr: EpisodeTrajectory, tau: float) -> Optional[float]:
    if tr.label_aki != 1 or tr.t_aki_hours is None or tr.n_postop_steps == 0:
        return None
    t_alert = first_alert_time(tr.postop_hours, tr.postop_probs, tau)
    if t_alert is None or t_alert >= tr.t_aki_hours:
        return None
    return float(tr.t_aki_hours - t_alert)


def select_aki_positive_case(
    trajectories: List[EpisodeTrajectory],
    tau: float,
    person_id: Optional[str] = None,
) -> EpisodeTrajectory:
    if person_id:
        for tr in trajectories:
            if str(tr.person_id) == str(person_id) and tr.label_aki == 1:
                return tr
        raise ValueError(f"AKI+ person_id={person_id} not found.")

    candidates: List[Tuple[float, EpisodeTrajectory]] = []
    for tr in trajectories:
        if tr.label_aki != 1 or tr.n_postop_steps == 0 or tr.t_aki_hours is None:
            continue
        lead = _lead_time_hours(tr, tau)
        if lead is None:
            continue
        t_alert = first_alert_time(tr.postop_hours, tr.postop_probs, tau)
        if t_alert is None:
            continue
        # Prefer illustrative lead: 24h–14d, high max risk
        score = float(lead) + 0.01 * float(np.max(tr.postop_probs))
        if 24 <= lead <= 336:
            score += 50.0
        candidates.append((score, tr))
    if not candidates:
        # fallback: any AKI+ with postop
        for tr in trajectories:
            if tr.label_aki == 1 and tr.n_postop_steps > 0:
                return tr
        raise ValueError("No AKI+ episode with postoperative steps found.")
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def select_aki_negative_case(
    trajectories: List[EpisodeTrajectory],
    tau: float,
    person_id: Optional[str] = None,
) -> EpisodeTrajectory:
    if person_id:
        for tr in trajectories:
            if str(tr.person_id) == str(person_id) and tr.label_aki == 0:
                return tr
        raise ValueError(f"AKI− person_id={person_id} not found.")

    candidates: List[Tuple[float, EpisodeTrajectory]] = []
    for tr in trajectories:
        if tr.label_aki != 0 or tr.n_postop_steps == 0:
            continue
        max_p = float(np.max(tr.postop_probs))
        t_alert = first_alert_time(tr.postop_hours, tr.postop_probs, tau)
        # Prefer: never alerted, low max risk, long follow-up
        if t_alert is None and max_p < tau:
            score = 100.0 - max_p + 0.001 * tr.postop_patient_days
            candidates.append((score, tr))
    if not candidates:
        # fallback: min max online prob among AKI−
        neg = [tr for tr in trajectories if tr.label_aki == 0 and tr.n_postop_steps > 0]
        if not neg:
            raise ValueError("No AKI− episode with postoperative steps found.")
        return min(neg, key=lambda t: float(np.max(t.postop_probs)))
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


MODALITY_COLORS = {
    "condition": "#2ca02c",
    "lab": "#1f77b4",
    "med": "#ff7f0e",
    "procedure": "#9467bd",
    "surgery": "#8c564b",
    "static": "#7f7f7f",
}


def _postop_tokens_from_seq(seq: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [tok for tok in seq if int(tok.get("is_postop", 0)) == 1]


def _short_event_label(tok: Dict[str, Any], max_len: int = 16) -> str:
    mod = str(tok.get("modality", "ev"))[:3]
    code = str(tok.get("raw_code", ""))
    if "|" in code:
        code = code.split("|", 1)[-1]
    code = code.replace("SNOMED", "").replace("RXCUI", "").strip(": ")
    if len(code) > max_len:
        code = code[: max_len - 2] + ".."
    return f"{mod}:{code}" if code else mod


def _threshold_crossings(
    hours: np.ndarray, probs: np.ndarray, tau: float
) -> List[Tuple[int, str]]:
    """Return (index, 'up'|'down') for each threshold crossing between consecutive steps."""
    if len(hours) == 0:
        return []
    out: List[Tuple[int, str]] = []
    if probs[0] >= tau:
        out.append((0, "up"))
    for i in range(1, len(probs)):
        if probs[i - 1] < tau and probs[i] >= tau:
            out.append((i, "up"))
        elif probs[i - 1] >= tau and probs[i] < tau:
            out.append((i, "down"))
    return out


def _prob_at_hour(hours: np.ndarray, probs: np.ndarray, t: float) -> float:
    if len(hours) == 0:
        return 0.0
    i = int(np.argmin(np.abs(hours - t)))
    return float(probs[i])


def _collect_axis_annotations(
    hours: np.ndarray,
    probs: np.ndarray,
    postop_toks: List[Dict[str, Any]],
    tau: float,
    t_alert: Optional[float],
    t_aki: Optional[float],
    max_labels: int = 12,
    min_sep_hours: float = 18.0,
) -> List[Dict[str, Any]]:
    """Milestones to draw on the risk panel x-axis (time, y, label, color, marker)."""
    entries: List[Dict[str, Any]] = []

    def add(t: float, headline: str, tok: Optional[Dict[str, Any]], priority: int, marker: str) -> None:
        if not np.isfinite(t):
            return
        y = _prob_at_hour(hours, probs, t)
        if tok is not None:
            body = _short_event_label(tok)
            label = f"{headline}\n{body}"
            color = MODALITY_COLORS.get(str(tok.get("modality", "")), "gray")
        else:
            label = headline
            color = "gray"
        entries.append(
            {
                "t": float(t),
                "y": y,
                "label": label,
                "color": color,
                "priority": priority,
                "marker": marker,
            }
        )

    for i, kind in _threshold_crossings(hours, probs, tau):
        tok = postop_toks[i] if i < len(postop_toks) else None
        if kind == "up":
            add(float(hours[i]), "Cross ↑τ", tok, priority=2, marker="^")
        else:
            add(float(hours[i]), "Cross ↓τ", tok, priority=3, marker="v")

    if t_alert is not None:
        i_alert = next((i for i, p in enumerate(probs) if p >= tau), 0)
        tok = postop_toks[i_alert] if i_alert < len(postop_toks) else None
        add(t_alert, "First alert", tok, priority=0, marker="*")

    if t_aki is not None and np.isfinite(t_aki):
        add(t_aki, "AKI onset", None, priority=0, marker="D")

    if len(probs) > 0:
        i_peak = int(np.argmax(probs))
        if i_peak > 0:
            tok = postop_toks[i_peak] if i_peak < len(postop_toks) else None
            add(float(hours[i_peak]), "Peak risk", tok, priority=4, marker="o")

    entries.sort(key=lambda e: e["t"])
    # De-duplicate nearby labels (keep higher priority = lower number)
    kept: List[Dict[str, Any]] = []
    for e in entries:
        if kept and abs(e["t"] - kept[-1]["t"]) < min_sep_hours:
            if e["priority"] < kept[-1]["priority"]:
                kept[-1] = e
            continue
        kept.append(e)
    return kept[:max_labels]


def _annotate_risk_axis(
    ax: Any,
    hours: np.ndarray,
    probs: np.ndarray,
    annotations: List[Dict[str, Any]],
) -> None:
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_min = float(np.min(probs)) if len(probs) else 0.0
    for e in annotations:
        t, y, label = e["t"], e["y"], e["label"]
        color = e.get("color", "gray")
        ax.axvline(t, color=color, linewidth=0.6, alpha=0.35, linestyle=":")
        ax.scatter([t], [y], c=color, s=42, zorder=6, edgecolors="k", linewidths=0.4)
        ax.plot([t], [y_min], "v", color=color, markersize=6, clip_on=False, zorder=5)
        ax.text(
            t,
            -0.14,
            label,
            transform=trans,
            rotation=42,
            ha="right",
            va="top",
            fontsize=6.5,
            color=color,
            linespacing=0.9,
        )


def plot_clinical_case_study(
    tr: EpisodeTrajectory,
    episode: Dict[str, Any],
    seq: List[Dict[str, Any]],
    out_path: Path,
    tau: float,
    attr_block: Optional[Dict[str, Any]] = None,
    dpi: int = 150,
) -> Dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib required for case study plots") from e

    hours = tr.postop_hours
    probs = tr.postop_probs
    t_alert = first_alert_time(hours, probs, tau) if len(hours) else None
    t_aki = tr.t_aki_hours if tr.label_aki == 1 else None
    lead = (t_aki - t_alert) if (t_aki is not None and t_alert is not None and t_alert < t_aki) else None

    if attr_block:
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3.0, 0.9, 2.0], "hspace": 0.32}
        )
        ax0, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(11, 5.8), gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.28}
        )
        ax0, ax2 = axes
        ax3 = None

    label_str = "AKI+" if tr.label_aki == 1 else "AKI−"
    fig.suptitle(
        f"Case study: person_id={tr.person_id} ({label_str}) | baseline p={tr.baseline_prob:.3f}",
        fontsize=11,
        y=0.98,
    )

    postop_toks = _postop_tokens_from_seq(seq)
    axis_ann = _collect_axis_annotations(
        hours, probs, postop_toks, tau, t_alert, t_aki, max_labels=12, min_sep_hours=18.0
    )

    # --- Risk trajectory + x-axis event labels at key times ---
    if len(hours):
        ax0.plot(hours, probs, color="C1", linewidth=1.8, label="Online AKI risk", zorder=2)
        ax0.scatter(hours, probs, c="C1", s=10, alpha=0.35, zorder=2)
    ax0.axhline(tau, color="gray", linestyle="--", linewidth=0.9, label=f"Alert threshold τ={tau}", zorder=1)
    ax0.axhline(tr.baseline_prob, color="C0", linestyle=":", linewidth=1.2, label="Baseline risk", zorder=1)
    _annotate_risk_axis(ax0, hours, probs, axis_ann)
    ax0.set_ylabel("Predicted AKI probability")
    y_top = min(1.02, max(1.0, float(np.max(probs)) * 1.12) if len(probs) else 1.0)
    ax0.set_ylim(-0.02, y_top)
    ax0.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)
    ax0.grid(True, alpha=0.25)
    subtitle = []
    if t_alert is not None:
        subtitle.append(f"T_alert={t_alert:.1f}h")
    if t_aki is not None:
        subtitle.append(f"T_AKI={t_aki:.1f}h")
    if lead is not None:
        subtitle.append(f"lead={lead:.1f}h")
    if subtitle:
        ax0.set_title(" | ".join(subtitle), fontsize=9)
    ax0.set_xlabel("Postoperative hours since surgery")
    plt.setp(ax0.get_xticklabels(), visible=True)

    # --- Summary text ---
    ax2.axis("off")
    max_p = float(np.max(probs)) if len(probs) else float("nan")
    lines = [
        f"Episode label: {label_str}",
        f"Baseline probability: {tr.baseline_prob:.4f}",
        f"Max online probability: {max_p:.4f}",
        f"Postoperative follow-up: {tr.postop_patient_days:.2f} patient-days",
        f"Postoperative event steps: {tr.n_postop_steps}",
        "",
        f"Alert rule: first postop step with p_online ≥ {tau}",
    ]
    if tr.label_aki == 1:
        if t_alert is not None and t_aki is not None and t_alert < t_aki:
            lines.append("Early warning: YES (alert before AKI onset)")
            lines.append(f"Lead time: {lead:.1f} hours ({lead/24:.1f} days)")
        elif t_alert is not None:
            lines.append("Early warning: NO (alert at or after AKI onset)")
        else:
            lines.append("Early warning: NO (no alert before end of trajectory)")
    else:
        if t_alert is not None:
            lines.append("False alert: YES (≥1 postop step above threshold)")
        else:
            lines.append("False alert: NO (online risk remained below threshold)")
    lines.append("")
    lines.append("Note: Online label at each step is episode-level AKI outcome (incidence framing).")
    ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    # --- Optional attribution ---
    meta: Dict[str, Any] = {
        "person_id": tr.person_id,
        "label_aki": tr.label_aki,
        "t_alert_hours": t_alert,
        "t_aki_hours": t_aki,
        "lead_time_hours": lead,
        "baseline_prob": tr.baseline_prob,
        "max_online_prob": max_p,
    }
    if attr_block and ax3 is not None:
        ax3.axis("off")
        online = attr_block.get("online_attribution", [])
        q_t = t_alert if t_alert is not None else (float(hours[-1]) if len(hours) else None)
        at_q = [r for r in online if q_t is not None and abs(r["query_time_hour"] - q_t) < 1e-3]
        if not at_q and online:
            last_q = max(r["query_time_hour"] for r in online)
            at_q = [r for r in online if abs(r["query_time_hour"] - last_q) < 1e-3]
        pos = sorted(at_q, key=lambda r: r["delta_logit"], reverse=True)[:8]
        neg_attr = sorted(at_q, key=lambda r: r["delta_logit"])[:8]
        alines = [
            "Model-attributed contributors (occlusion Δlogit; not causal):",
            f"Explanation time ≈ {q_t:.1f} h postop",
            "",
            "Risk-increasing:",
        ]
        for r in pos:
            alines.append(f"  + [{r['modality']}] {r['raw_code']}: {r['delta_logit']:.3f}")
        alines.append("")
        alines.append("Risk-decreasing:")
        for r in neg_attr:
            alines.append(f"  - [{r['modality']}] {r['raw_code']}: {r['delta_logit']:.3f}")
        ax3.text(0.02, 0.98, "\n".join(alines), va="top", ha="left", fontsize=8, family="monospace")
        meta["attribution_query_hour"] = q_t
    meta["axis_annotations"] = [{"t": e["t"], "label": e["label"]} for e in axis_ann]

    fig.subplots_adjust(top=0.93, bottom=0.20, left=0.08, right=0.96)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    train_args = ckpt.get("args", {}) or {}
    episodes = load_episode_jsonl(args.episode_path)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)

    loader = build_eval_dataloader(args.episode_path, code_vocab, lab_stats, num_med, args.batch_size, age_stats=age_stats)
    model = build_model_from_checkpoint(code_vocab, train_args, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)

    print(f"Building trajectories for {len(episodes)} episodes...")
    trajectories, ep_by_id = collect_trajectories(model, loader, device, episodes)

    tr_pos = select_aki_positive_case(trajectories, args.alert_tau, args.person_id_pos)
    tr_neg = select_aki_negative_case(trajectories, args.alert_tau, args.person_id_neg)
    print(f"AKI+ case: person_id={tr_pos.person_id}")
    print(f"AKI− case: person_id={tr_neg.person_id}")

    report: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path).resolve()),
        "alert_tau": args.alert_tau,
        "cases": {},
    }

    for tag, tr in (("aki_positive", tr_pos), ("aki_negative", tr_neg)):
        ep = ep_by_id[str(tr.person_id)]
        feat = episode_to_sequence_features(
            ep, code_vocab, add_base=True, lab_stats=lab_stats, age_stats=age_stats, num_med_exposure=num_med
        )
        attr_block = None
        if args.with_attribution:
            print(f"Attribution for {tag} person_id={tr.person_id} ...")
            attr_block = attributions_for_episode(
                model,
                ep,
                code_vocab,
                lab_stats,
                num_med,
                device,
                max_preop_occlusions=args.max_preop_occlusions,
                max_postop_query_steps=args.max_postop_query_steps,
            )
        out_png = out_dir / f"case_study_{tag}_person_{tr.person_id}.png"
        meta = plot_clinical_case_study(
            tr, ep, feat["sequence"], out_png, args.alert_tau, attr_block=attr_block, dpi=args.dpi
        )
        report["cases"][tag] = meta
        print(f"Wrote {out_png}")

    json_path = out_dir / "case_study_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
