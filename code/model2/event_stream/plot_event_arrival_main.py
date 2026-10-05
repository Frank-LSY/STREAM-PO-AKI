#!/usr/bin/env python3
"""Plot event-arrival main figures: case staircases (hero) + symbolic rank + concordance."""
from __future__ import annotations

import csv
import json
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import fig_style
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
FIG = ROOT / "result/figures/event_arrival"
POOL = ROOT / "result/event_stream/transformer/event_arrival_pooled"
STEPS = ROOT / "result/event_stream/transformer/event_arrival_fold1/event_arrival_steps.csv"
# Order-robust case-study steps (written by evaluate_event_arrival --person_ids …).
CASE_STEPS = ROOT / "result/event_stream/transformer/event_arrival_cases_order_robust/event_arrival_steps.csv"
CF = ROOT / "result/event_stream/transformer/contribution_5fold_full"
RED, BLUE = "#c0392b", "#1f6cb0"
# Desaturated force-bar fills (non-emphasized)
RED_FADE, BLUE_FADE = "#e8b4ae", "#b7cfe0"
RISK_THR = 0.5
POSTOP_7D_H = 168.0
MIN_ARRIVAL_PATIENTS = 50
T_CRIT_5FOLD = 2.776  # two-sided 95% CI across 5 fold means (df=4)
# Shared risk axis for AKI+/AKI− case panels (room past 1 for stacked force bars).
RISK_XLIM = (-0.02, 1.20)
HORIZON_H = 48.0
# PO-AKI− panel window; risk stays < 0.5 through 7 d in all 5 folds.
NEG_MONITOR_END_H = 48.0
MAX_ANNOT = 5
# Clear trend narratives (independent axes; not max-|Δz| chaos).
# Prefer AKI onset within model-stable 7d (ideally ≤12h / 2d) on the test split.
# Case-study person_ids are site data and are not distributed; set them via environment variables.
CURATED_POS = int(os.environ.get("STREAM_CASE_POS_ID", "-1"))
CURATED_NEG = int(os.environ.get("STREAM_CASE_NEG_ID", "-2"))
LEGACY_POS = int(os.environ.get("STREAM_CASE_LEGACY_POS_ID", "-3"))
EPISODES_PKL = ROOT / "data/merged_7d_window_clm.pkl"
# Prefer dedup4 test JSONL (matches event_arrival evaluation) over the merged pickle.
EPISODES_JSONL = ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"
PROC_NAMES = ROOT / "data/snomed_procedure_names.json"
CASE_META_CACHE = FIG / "case_clinical_meta.json"

_CASE_META: dict[int, dict] | None = None


def _short_proc_name(code: str, names: dict) -> str:
    raw = str(names.get(code) or names.get(code.split("|")[-1]) or code)
    s = " ".join(raw.split())
    for suf in (" (procedure)", " (Procedure)"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def _format_aki_time(hours: float | None) -> str | None:
    if hours is None or not np.isfinite(hours):
        return None
    if hours >= 72.0:
        return f"PO-AKI at {hours / 24.0:.1f} d"
    return f"PO-AKI at {hours:.1f} h"


def _meta_from_episode(ep: dict, names: dict) -> dict:
    from datetime import datetime

    procs = ep.get("procedure_tokens") or []
    codes = [str(p.get("code")) for p in procs if p.get("code")]
    surgery = _short_proc_name(codes[0], names) if codes else "Unknown surgery"
    aki_h = None
    t0, aki = ep.get("t0"), ep.get("aki_time")
    if t0 and aki:
        try:
            t0d = datetime.fromisoformat(str(t0).replace(" ", "T"))
            akid = datetime.fromisoformat(str(aki).replace(" ", "T"))
            aki_h = (akid - t0d).total_seconds() / 3600.0
        except Exception:
            aki_h = None
    return {
        "surgery": surgery,
        "procedure_code": codes[0] if codes else None,
        "aki_hours": aki_h,
        "label_aki": int(ep.get("label_aki", 0)),
    }


def _load_case_clinical_meta(pids: list[int] | None = None) -> dict[int, dict]:
    """Surgery name + AKI onset hours from test JSONL (preferred) or pickle (cached)."""
    global _CASE_META
    FIG.mkdir(parents=True, exist_ok=True)
    want = set(int(p) for p in (pids or []))
    cached: dict[int, dict] = {}
    if CASE_META_CACHE.exists():
        try:
            raw = json.loads(CASE_META_CACHE.read_text())
            cached = {int(k): v for k, v in raw.items()}
        except Exception:
            cached = {}
    # Invalidate stale cache entries that lack aki_hours for requested AKI+ pids.
    if want and want.issubset(cached.keys()):
        _CASE_META = cached
        return cached

    names = json.loads(PROC_NAMES.read_text()) if PROC_NAMES.exists() else {}
    need = set(want) if want else None

    if EPISODES_JSONL.exists():
        with EPISODES_JSONL.open() as f:
            for line in f:
                ep = json.loads(line)
                try:
                    pid = int(ep.get("person_id"))
                except Exception:
                    continue
                if need is not None and pid not in need:
                    continue
                cached[pid] = _meta_from_episode(ep, names)
                if need is not None and need.issubset(cached.keys()):
                    break
    elif EPISODES_PKL.exists():
        import pickle

        with EPISODES_PKL.open("rb") as f:
            episodes = pickle.load(f)
        for ep in episodes:
            try:
                pid = int(ep.get("person_id"))
            except Exception:
                continue
            if need is not None and pid not in need:
                continue
            cached[pid] = _meta_from_episode(ep, names)
            if need is not None and need.issubset(cached.keys()):
                break

    CASE_META_CACHE.write_text(json.dumps({str(k): v for k, v in cached.items()}, indent=2))
    _CASE_META = cached
    return cached


def _outcome_label(pref: str, pid: int) -> str:
    """Formal case role for figure titles."""
    p = str(pref)
    if p.startswith("AKI+") or "positive" in p.lower():
        return "PO-AKI positive"
    if p.startswith("AKI") or "negative" in p.lower() or "\u2212" in p:
        return "PO-AKI negative"
    meta = (_CASE_META or {}).get(int(pid), {})
    return "PO-AKI positive" if int(meta.get("label_aki", 0)) == 1 else "PO-AKI negative"


# Curated panel subtitle stems; risk range is filled from the plotted path.
_PANEL_SUBTITLE_STEM: dict[int, str] = {
    CURATED_POS: "Laparoscopic appendectomy | PO-AKI at 15.4 h",
    CURATED_NEG: "Exploratory laparotomy | No PO-AKI within 30 days",
}

_PANEL_ROLE: dict[int, str] = {
    CURATED_POS: "PO-AKI positive episode",
    CURATED_NEG: "PO-AKI negative episode",
}


def _case_title_parts(
    pref: str,
    pid: int,
    baseline: float,
    final: float,
    panel: str | None = None,
) -> tuple[str, str]:
    meta = (_CASE_META or _load_case_clinical_meta([pid])).get(int(pid), {})
    stem = _PANEL_SUBTITLE_STEM.get(int(pid))
    role = _PANEL_ROLE.get(int(pid)) or _outcome_label(pref, pid)
    risk = f"Predicted PO-AKI probability: {baseline:.2f} \u2192 {final:.2f}"
    if stem:
        head = fig_style.panel_label(panel, role)
        return head, f"{stem} | {risk}"
    surgery = meta.get("surgery") or "Surgery"
    details = [surgery]
    if role.endswith("positive") or role.endswith("positive episode") or int(meta.get("label_aki", 0)) == 1:
        aki_s = _format_aki_time(meta.get("aki_hours"))
        if aki_s:
            details.append(aki_s)
    details.append(risk)
    head = f"{panel}  {role}" if panel else role
    return head, "  \u00b7  ".join(details)


def _case_title(
    pref: str,
    pid: int,
    baseline: float,
    final: float,
    panel: str | None = None,
) -> str:
    head, detail = _case_title_parts(pref, pid, baseline, final, panel=panel)
    return f"{head}\n{detail}"


def _apply_case_title(ax, pref: str, pid: int, baseline: float, final: float,
                      panel: str | None = None) -> None:
    """Two-line panel title above the axes (letter, then curated subtitle)."""
    head, detail = _case_title_parts(pref, pid, baseline, final, panel=panel)
    # Avoid set_title + floating text (they share the same band and overlap).
    ax.set_title("")
    ax.text(
        0.0, 1.055, head,
        transform=ax.transAxes,
        ha="left", va="bottom",
        **fig_style.panel_kw(ax.figure),
        clip_on=False, zorder=30,
    )
    ax.text(
        0.0, 1.012, detail,
        transform=ax.transAxes,
        ha="left", va="bottom",
        **fig_style.subtitle_kw(ax.figure),
        clip_on=False, zorder=30,
    )


def _names():
    ccs = {r["ccs_code"]: r["definition"] for r in csv.DictReader(open(ROOT / "data/ccs_codes.csv"))}
    rx = dict(l.strip().split(",", 1) for l in open(ROOT / "data/rxcui_to_name.csv") if "," in l)
    labn = json.load(open(ROOT / "data/lab_concept_names.json"))

    def nm(code: str, w: int | None = 28) -> str:
        p = code.split("|")
        if code.startswith("CCS|"):
            s = ccs.get(p[1], code)
        elif code.startswith("RXCUI|"):
            s = rx.get(code, code)
        elif code.startswith("LAB|"):
            s = labn.get(p[1], code)
        else:
            s = code
        s = " ".join(str(s).split())
        if str(s).upper() == "PROPRIETARY":
            s = code
        if w is None or w <= 0:
            return s
        return s if len(s) <= w else s[: w - 1] + "\u2026"

    return nm


def _add_fold_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Attach 95% CI bounds from fold-level SD (across 5 CV folds)."""
    out = df.copy()
    if "ci_lo" in out.columns and "ci_hi" in out.columns:
        return out
    nf = out.get("n_folds", pd.Series(5, index=out.index)).to_numpy(dtype=float)
    sd = out.get("fold_sd", pd.Series(0.0, index=out.index)).to_numpy(dtype=float)
    se = sd / np.sqrt(np.clip(nf, 1.0, None))
    half = T_CRIT_5FOLD * se
    mu = out["mean_delta_logit"].to_numpy(dtype=float)
    out["ci_lo"] = mu - half
    out["ci_hi"] = mu + half
    return out


def _pool_arrival_code_table(
    *,
    horizon_h: float | None = None,
    min_patients: int = MIN_ARRIVAL_PATIENTS,
) -> pd.DataFrame:
    """Pool per-fold arrival a_i; optional postop ``horizon_h`` filter from steps.

    ``delta_logit`` in steps is order-robust ``a_i`` (Δz or φ̂).
    """
    frames: list[pd.DataFrame] = []
    for f in range(1, 6):
        fold_dir = ROOT / f"result/event_stream/transformer/event_arrival_fold{f}"
        if horizon_h is None:
            p = fold_dir / "event_arrival_by_code.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            t = pd.read_csv(p)
        else:
            steps_p = fold_dir / "event_arrival_steps.csv"
            if not steps_p.exists():
                raise FileNotFoundError(steps_p)
            steps = pd.read_csv(
                steps_p,
                usecols=["person_id", "modality", "raw_code", "time_hour", "delta_logit"],
            )
            steps = steps[(steps.time_hour >= 0) & (steps.time_hour < float(horizon_h))]
            pc = (
                steps.groupby(["person_id", "modality", "raw_code"], as_index=False)["delta_logit"]
                .mean()
            )
            t = (
                pc.groupby(["modality", "raw_code"], as_index=False)
                .agg(
                    n_patients=("person_id", "nunique"),
                    mean_delta_logit=("delta_logit", "mean"),
                    mean_abs_delta_logit=("delta_logit", lambda s: float(np.abs(s).mean())),
                )
            )
        t["fold"] = f
        frames.append(t)

    df = pd.concat(frames, ignore_index=True)
    rows: list[dict] = []
    for (mod, code), g in df.groupby(["modality", "raw_code"]):
        if len(g) < 5:
            continue
        means = g["mean_delta_logit"].to_numpy(dtype=float)
        ns = g["n_patients"].to_numpy(dtype=float)
        w = ns / ns.sum()
        pooled = float(np.sum(w * means))
        signs = np.sign(means)
        agree = int(np.all(signs == signs[0]) and signs[0] != 0)
        rows.append(
            {
                "modality": mod,
                "raw_code": code,
                "n_folds": int(len(g)),
                "n_patients_max": int(ns.max()),
                "n_patients_mean": float(ns.mean()),
                "mean_delta_logit": pooled,
                "fold_min": float(means.min()),
                "fold_max": float(means.max()),
                "fold_sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "folds_agreeing_sign": agree,
                "mean_abs_delta_logit": float(
                    np.sum(w * g["mean_abs_delta_logit"].to_numpy(dtype=float))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out = out[out.n_patients_max >= min_patients].copy()
    out = out.sort_values("mean_delta_logit", key=lambda s: s.abs(), ascending=False)
    return _add_fold_ci(out)


RANK_DISPLAY_NAMES = {
    "CCS|210": "SLE and connective\ntissue disorders",
    "CCS|182": "Hemorrhage during\npregnancy",
    "CCS|99": "Hypertension with\ncomplications",
    "CCS|161": "Other kidney and\nureter diseases",
    "CCS|50": "Diabetes with\ncomplications",
    "CCS|195": "Other complications\nof birth",
    "CCS|176": "Contraceptive and\nprocreative management",
    "CCS|232": "Sprains and strains",
    "CCS|115": "Aortic, peripheral and\nvisceral aneurysms",
    "CCS|184": "Early or threatened\nlabor",
    "CCS|208": "Acquired foot\ndeformities",
    "CCS|100": "Acute myocardial\ninfarction",
    "CCS|257": "Other aftercare",
    "CCS|55": "Fluid and electrolyte\ndisorders",
    "CCS|64": "Other hematologic\nconditions",
    "LAB|3004249": "Systolic blood\npressure",
}

# RxNorm names this retired MMSL concept "PROPRIETARY"; it has no ingredient mapping.
RANK_EXCLUDE_CODES = {"RXCUI|219376"}


def _rank_label(nm, code: str, width: int = 24) -> str:
    if code in RANK_DISPLAY_NAMES:
        return RANK_DISPLAY_NAMES[code]
    s = nm(code, None)
    if s.islower():
        s = s[:1].upper() + s[1:]
    return textwrap.fill(s, width=width, break_long_words=False)


def plot_symbolic_rank(
    nm,
    rep: pd.DataFrame,
    *,
    out_name: str = "arrival_rank_symbolic.png",
    suptitle: str = "Event-arrival code ranking (secondary / symbolic)",
    xlab: str = "Pooled arrival \u0394logit (5-fold, sign-replicated)",
    left_title: str = "Symbolic: top increasing arrivals",
    right_title: str = "Symbolic: top decreasing arrivals",
    panel_letters: tuple[str, str] | None = None,
    top_n: int = 10,
) -> Path:
    rep = _add_fold_ci(rep)
    rep = rep[~rep.raw_code.isin(RANK_EXCLUDE_CODES)]
    pos = rep[rep.mean_delta_logit > 0].nlargest(top_n, "mean_delta_logit")
    neg = rep[rep.mean_delta_logit < 0].nsmallest(top_n, "mean_delta_logit")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    letters = panel_letters or ("", "")
    for ax, sub, color, title, letter in (
        (axes[0], pos.iloc[::-1], RED, left_title, letters[0]),
        (axes[1], neg.iloc[::-1], BLUE, right_title, letters[1]),
    ):
        y = np.arange(len(sub), dtype=float)
        vals = sub.mean_delta_logit.to_numpy(dtype=float)
        err_lo = vals - sub.ci_lo.to_numpy(dtype=float)
        err_hi = sub.ci_hi.to_numpy(dtype=float) - vals
        ax.barh(
            y, vals,
            color=color, alpha=0.78, height=0.62,
            edgecolor="white", linewidth=0.4, zorder=2,
        )
        ax.errorbar(
            vals, y,
            xerr=[err_lo, err_hi],
            fmt="none",
            ecolor="0.40",
            lw=0.95,
            capsize=2.2,
            capthick=0.9,
            alpha=0.85,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{_rank_label(nm, c)} (n={int(n):,})" for c, n in zip(sub.raw_code, sub.n_patients_max)],
            fontsize=8.0,
            linespacing=1.05,
        )
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.axvline(0, color="0.55", lw=0.7, zorder=1)
        ax.set_xlabel(xlab, fontsize=9.5, color="0.2")
        if letter:
            fig_style.panel_title(ax, letter, title, pad=8)
        else:
            fig_style.panel_title(ax, None, title, pad=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.75")
        ax.spines["bottom"].set_color("0.75")
        ax.tick_params(axis="x", labelsize=8, colors="0.3")
        ax.tick_params(axis="y", length=0, pad=3)
        ax.grid(axis="x", alpha=0.18, lw=0.7)
        ax.set_axisbelow(True)
    fig_style.add_suptitle(fig, suptitle, y=0.995)
    fig.tight_layout(w_pad=2.2, rect=(0, 0, 1, 0.965))
    out = FIG / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_concordance(nm, rep: pd.DataFrame) -> Path:
    online = pd.read_csv(CF / "crossfold_online.csv")
    post = ["postop_0_6h", "postop_6_24h", "postop_24_72h", "postop_72h_7d"]
    sub = online[
        online.time_bin.isin(post)
        & online.replicated
        & online.modality.isin(["condition", "med", "lab"])
    ]
    del_rows = []
    for code, g in sub.groupby("raw_code"):
        i = g.pooled_delta_logit.abs().idxmax()
        r = g.loc[i]
        del_rows.append(
            {"raw_code": code, "modality": r.modality, "deletion_peak": float(r.pooled_delta_logit)}
        )
    dele = pd.DataFrame(del_rows)
    m = rep.merge(dele, on=["raw_code", "modality"], how="inner")
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)
    lim = float(max(m.mean_delta_logit.abs().max(), m.deletion_peak.abs().max()) * 1.1)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="0.7", lw=1.0, zorder=1)
    ax.scatter(
        m.deletion_peak, m.mean_delta_logit,
        s=18 + np.sqrt(m.n_patients_max) * 1.8,
        c="#34495e", alpha=0.55, edgecolors="white", lw=0.35, zorder=3,
    )
    score = m.mean_delta_logit.abs() + m.deletion_peak.abs()
    for _, r in m.loc[score.nlargest(8).index].iterrows():
        ax.annotate(
            nm(r.raw_code, 22), (r.deletion_peak, r.mean_delta_logit),
            fontsize=7, xytext=(3, 3), textcoords="offset points", color="0.2",
        )
    ax.set_xlabel("Grouped-deletion peak \u0394logit (postop)")
    ax.set_ylabel("Arrival pooled \u0394logit (5-fold replicated)")
    ax.set_title("Reliance vs update (deletion \u2260 arrival)", fontsize=11.5, weight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    out = FIG / "arrival_vs_deletion.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close()
    return out


def _pid(x) -> int:
    return int(float(x))


def _readable(code: str) -> bool:
    s = str(code)
    if s in ("MED_OTHERS", "COND_OTHERS", "LAB_OTHERS"):
        return False
    if s.startswith("RXCUI|") and "PROPRIETARY" in s.upper():
        return False
    return True


def _pick_annotations(d: pd.DataFrame, k: int = MAX_ANNOT, min_sep_h: float = 0.75) -> pd.DataFrame:
    """Largest |Δz| arrivals, preferring temporal spread and readable codes."""
    ranked = d.reindex(d.delta_logit.abs().sort_values(ascending=False).index)
    picked = []
    times: list[float] = []
    for _, r in ranked.iterrows():
        if not _readable(r.raw_code):
            continue
        t = float(r.time_hour)
        if any(abs(t - tt) < min_sep_h for tt in times):
            continue
        picked.append(r)
        times.append(t)
        if len(picked) >= k:
            break
    if len(picked) < max(3, k // 2):
        for _, r in ranked.iterrows():
            if not _readable(r.raw_code):
                continue
            if any(r.name == p.name for p in picked):
                continue
            picked.append(r)
            if len(picked) >= k:
                break
    return pd.DataFrame(picked) if picked else ranked.head(k)


def _path_features(g: pd.DataFrame) -> pd.Series:
    p = g.prob.to_numpy(dtype=float)
    t = g.time_hour.to_numpy(dtype=float)
    dz = g.delta_logit.to_numpy(dtype=float)
    tv = float(np.abs(np.diff(p)).sum()) if len(p) > 1 else 0.0
    rise = float(p[-1] - p[0])
    peak_to_end = float(p.max() - p[-1])
    return pd.Series(
        {
            "n": len(g),
            "start_prob": float(p[0]),
            "end_prob": float(p[-1]),
            "prob_range": float(p.max() - p.min()),
            "tv": tv,
            "rise": rise,
            "peak_to_end": peak_to_end,
            "max_jump": float(np.abs(dz).max()) if len(dz) else 0.0,
            "t_span": float(t.max() - t.min()) if len(t) else 0.0,
            "sat": float(((p < 0.01) | (p > 0.99)).mean()),
        }
    )


def _load_case_candidates(steps_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "person_id", "label_aki", "step", "raw_code", "modality",
        "time_hour", "delta_logit", "prob", "prev_prob",
    ]
    chunks = []
    for ch in pd.read_csv(steps_path, usecols=usecols, chunksize=400_000):
        ch = ch[(ch.time_hour >= 0) & (ch.time_hour <= HORIZON_H)].copy()
        if len(ch):
            ch["person_id"] = ch["person_id"].map(_pid)
            chunks.append(ch)
    early = pd.concat(chunks, ignore_index=True)
    agg = (
        early.sort_values(["person_id", "time_hour", "step"])
        .groupby(["person_id", "label_aki"], as_index=False)
        .apply(_path_features, include_groups=False)
        .reset_index(drop=True)
    )
    return early, agg


def select_cases(early: pd.DataFrame, agg: pd.DataFrame) -> tuple[int, int]:
    import os

    env_pos = os.environ.get("CASE_POS")
    env_neg = os.environ.get("CASE_NEG")
    if env_pos and env_neg:
        return int(env_pos), int(env_neg)

    have = set(agg.person_id.astype(int))
    if CURATED_POS in have and CURATED_NEG in have:
        return CURATED_POS, CURATED_NEG

    pos = agg[
        (agg.label_aki == 1)
        & (agg.n.between(12, 60))
        & (agg.end_prob >= 0.55)
        & (agg.rise >= 0.35)
        & (agg.prob_range >= 0.30)
        & (agg.t_span >= 6.0)
        & (agg.sat < 0.55)
        & (agg.tv / agg.prob_range.clip(lower=0.2) < 12)
    ].copy()
    pos["score"] = pos.rise + 0.35 * pos.prob_range + 0.02 * pos.t_span - 0.08 * (pos.tv / pos.prob_range.clip(lower=0.2))
    pos = pos.sort_values("score", ascending=False)

    aki_pos = None
    for _, row in pos.head(40).iterrows():
        pid = _pid(row.person_id)
        d = early[early.person_id == pid]
        top = _pick_annotations(d)
        if len(top) >= 3 and all(_readable(c) for c in top.raw_code):
            aki_pos = pid
            break
    if aki_pos is None:
        if pos.empty:
            raise RuntimeError("no AKI+ case candidates")
        aki_pos = _pid(pos.iloc[0].person_id)

    neg = agg[
        (agg.label_aki == 0)
        & (agg.n.between(12, 60))
        & (agg.end_prob <= 0.25)
        & (agg.peak_to_end >= 0.30)
        & (agg.start_prob >= 0.35)
        & (agg.t_span >= 8.0)
        & (agg.sat < 0.55)
        & (agg.tv / agg.prob_range.clip(lower=0.2) < 12)
    ].copy()
    neg["score"] = (
        neg.peak_to_end
        + 0.25 * neg.prob_range
        + 0.02 * neg.t_span
        - 0.08 * (neg.tv / neg.prob_range.clip(lower=0.2))
    )
    neg = neg.sort_values("score", ascending=False)

    aki_neg = None
    for _, row in neg.head(40).iterrows():
        pid = _pid(row.person_id)
        d = early[early.person_id == pid]
        top = _pick_annotations(d)
        if len(top) >= 3 and all(_readable(c) for c in top.raw_code):
            aki_neg = pid
            break
    if aki_neg is None:
        if neg.empty:
            raise RuntimeError("no AKI- case candidates")
        aki_neg = _pid(neg.iloc[0].person_id)
    return aki_pos, aki_neg


def _prepare_case_frame(
    d: pd.DataFrame,
    through_h: float | None = None,
) -> pd.DataFrame:
    """Sort arrivals; keep clock time primary. Crop to active postop window.

    If ``through_h`` is set (e.g. PO-AKI diagnosis hour), keep all events up to
    that clock time so the case panel is not truncated before the clinical endpoint.
    """
    out = d.sort_values(["time_hour", "step"]).reset_index(drop=True).copy()
    if through_h is not None and np.isfinite(float(through_h)):
        out = out[out.time_hour <= float(through_h) + 0.05].reset_index(drop=True)
    out["ord"] = np.arange(1, len(out) + 1)
    # Merge near-simultaneous arrivals into one clock-time force stack (~6 min bins).
    out["t_key"] = np.round(out.time_hour.to_numpy(dtype=float), 1)
    p = out.prob.to_numpy(dtype=float)
    if len(p) > 2:
        dp = np.abs(np.diff(p))
        c = np.cumsum(dp)
        tot = float(c[-1]) if float(c[-1]) > 0 else 1.0
        i90 = int(np.searchsorted(c, 0.90 * tot))
        i_end = min(len(out) - 1, i90 + max(3, len(out) // 10))
    else:
        i_end = len(out) - 1
    i_end = max(i_end, min(len(out) - 1, 8))
    if through_h is not None and np.isfinite(float(through_h)):
        # +0.05 h so a rounded end like 653.5 keeps the ~653.50 event cluster.
        keep = np.where(out.time_hour.to_numpy(dtype=float) <= float(through_h) + 0.05)[0]
        if len(keep):
            i_end = max(i_end, int(keep[-1]))
    return out.iloc[: i_end + 1].copy()


def _case_through_h(pid: int | None) -> float | None:
    """Clock hour that the case panel must reach (AKI+ diagnosis / AKI− last event)."""
    if pid is None:
        return None
    pid = int(pid)
    if pid == CURATED_NEG:
        return float(NEG_MONITOR_END_H)
    meta = (_CASE_META or {}).get(pid, {})
    aki_h = meta.get("aki_hours")
    if aki_h is not None and np.isfinite(float(aki_h)):
        return float(aki_h)
    if pid == CURATED_POS:
        return 8.0
    return None


def _batch_ends(d: pd.DataFrame) -> pd.DataFrame:
    """One row per clock-time bin: risk after all arrivals in the bin."""
    rows = []
    for t_key, g in d.groupby("t_key", sort=True):
        g = g.sort_values(["time_hour", "step"])
        z0 = float(g.prev_logit.iloc[0]) if "prev_logit" in g.columns else None
        if (
            z0 is not None
            and "delta_logit" in g.columns
            and "attribution_mode" in g.columns
            and (g.attribution_mode.astype(str) == "shapley_mc").any()
        ):
            z1 = z0 + float(g.delta_logit.sum())
            p0 = float(1.0 / (1.0 + np.exp(-z0)))
            p1 = float(1.0 / (1.0 + np.exp(-z1)))
            net = float(g.delta_logit.sum())
        else:
            p0 = float(g.prev_prob.iloc[0])
            p1 = float(g.prob.iloc[-1])
            net = float(g.delta_logit.sum()) if "delta_logit" in g.columns else float(p1 - p0)
        n_tie = int(g.tie_size.max()) if "tie_size" in g.columns else 1
        rows.append(
            {
                "t_key": float(t_key),
                "time_hour": float(t_key),
                "prob": p1,
                "prev_prob": p0,
                "net_delta": net,
                "n_events": int(len(g)),
                "max_tie_size": n_tie,
                "has_shapley": bool(
                    "attribution_mode" in g.columns
                    and (g.attribution_mode.astype(str) == "shapley_mc").any()
                ),
            }
        )
    return pd.DataFrame(rows)


def _time_xlim(d: pd.DataFrame) -> tuple[float, float]:
    t0 = float(d.time_hour.min())
    t1 = float(d.time_hour.max())
    span = max(t1 - t0, 1.0)
    return max(-0.3, t0 - 0.06 * span), t1 + max(0.8, 0.08 * span)


def _prob_ylim(vals: np.ndarray) -> tuple[float, float]:
    p0, p1 = float(np.min(vals)), float(np.max(vals))
    pr = max(p1 - p0, 0.08)
    pad = max(0.035, 0.10 * pr)
    y0, y1 = max(0.0, p0 - pad), min(1.0, p1 + pad)
    if y1 - y0 < 0.22:
        mid = 0.5 * (y0 + y1)
        y0, y1 = max(0.0, mid - 0.11), min(1.0, mid + 0.11)
    return y0, y1


def _draw_traj_panel(ax, d: pd.DataFrame, batches: pd.DataFrame, pref: str, pid: int) -> None:
    """Subplot 1: clean prediction trajectory vs postop time."""
    label = int(d.label_aki.iloc[0])
    color = RED if label == 1 else "#2c3e50"

    baseline = float(batches.iloc[0].prev_prob)
    t_first = float(batches.iloc[0].time_hour)
    # Start at 0h baseline, then staircase through event times
    y_cursor = baseline
    ts = [0.0, t_first]
    ps = [baseline, baseline]
    for _, b in batches.iterrows():
        t = float(b.time_hour)
        ts.extend([t, t])
        ps.extend([y_cursor, float(b.prob)])
        y_cursor = float(b.prob)

    ax.plot(ts, ps, color=color, lw=2.1, solid_capstyle="butt", zorder=3)
    ax.scatter(
        batches.time_hour, batches.prob,
        s=30, c=color, zorder=4, edgecolors="white", lw=0.5,
    )
    # Mark 0h baseline
    ax.scatter(
        [0.0], [baseline],
        s=36, c="0.45", zorder=4, edgecolors="white", lw=0.6,
    )

    t0, t1 = _time_xlim(d)
    ax.set_xlim(min(t0, -0.15), t1)
    all_y = np.concatenate([
        [baseline],
        batches.prev_prob.to_numpy(dtype=float),
        batches.prob.to_numpy(dtype=float),
    ])
    ax.set_ylim(*_prob_ylim(all_y))
    ax.set_ylabel("Predicted PO-AKI probability")
    ax.set_xlabel("Hours after surgery")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        f"{pref}  \u00b7  person {pid}  \u00b7  trajectory  \u00b7  "
        f"probability {baseline:.2f}\u2192{float(batches.iloc[-1].prob):.2f}",
        fontsize=11.0, weight="bold", loc="left",
    )


def _label_text(name: str, code: str) -> str:
    """Single-line label: code:description."""
    name = " ".join(str(name).split())
    code = str(code)
    if name == code:
        return code
    return f"{code}:{name}"


def _est_label_width(text: str, fontsize: float = 6.6) -> float:
    """Rough data-unit width for a callout (risk axis ~0–1)."""
    nchars = len(text)
    # Slightly generous so near-neighbors take separate vertical tiers.
    return 0.0105 * nchars * (fontsize / 6.6) + 0.032


def _layout_force_segments(g: pd.DataFrame, p_old: float) -> list[dict]:
    """Place force segments for one clock-time column (along risk / y).

    If both positive and negative arrivals exist: stack negatives below
    ``p_old`` and positives above (largest |Δlogit| nearest to p_old), flush.
    Otherwise keep arrival-order chaining (prev→new). Separation between
    abutting bars is drawn as a thick white edge at render time.
    """
    raw: list[dict] = []
    for _, r in g.iterrows():
        x0 = float(r.prev_prob)
        x1 = float(r.prob)
        w = abs(x1 - x0)
        if w < 1e-4:
            w = 1e-3
        raw.append(
            {
                "x0": x0,
                "x1": x1,
                "w": w,
                "dz": float(r.delta_logit),
                "code": str(r.raw_code),
                "modality": str(r.modality) if "modality" in g.columns else "",
                "pos": float(r.delta_logit) >= 0,
            }
        )

    has_pos = any(s["pos"] for s in raw)
    has_neg = any(not s["pos"] for s in raw)
    segs: list[dict] = []

    if has_pos and has_neg:
        negs = sorted(
            [s for s in raw if not s["pos"]],
            key=lambda s: abs(s["dz"]), reverse=True,
        )
        poss = sorted(
            [s for s in raw if s["pos"]],
            key=lambda s: abs(s["dz"]), reverse=True,
        )
        cursor = float(p_old)
        for s in negs:
            hi = cursor
            lo = cursor - s["w"]
            segs.append({**s, "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi)})
            cursor = lo
        cursor = float(p_old)
        for s in poss:
            lo = cursor
            hi = cursor + s["w"]
            segs.append({**s, "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi)})
            cursor = hi
    else:
        for s in raw:
            lo, hi = (s["x0"], s["x1"]) if s["x1"] >= s["x0"] else (s["x1"], s["x0"])
            if hi - lo < 1e-4:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - 5e-4, mid + 5e-4
            segs.append({**s, "lo": lo, "hi": hi, "mid": 0.5 * (lo + hi), "w": hi - lo})
    return segs


def _est_label_height(fontsize: float = 6.4) -> float:
    """Rough data-unit height for a callout on the risk axis."""
    return 0.034 * (fontsize / 6.4)


def _resolve_vertical_label_positions(
    ideal: list[float],
    min_sep: float,
    y_lo: float,
    y_hi: float,
) -> list[float]:
    """Pack label centers vertically while staying close to segment mids."""
    n = len(ideal)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: ideal[i])
    ys = [float(ideal[i]) for i in order]
    out = [ys[0]]
    for j in range(1, n):
        out.append(max(ys[j], out[-1] + min_sep))
    pad = 0.02
    if out[-1] > y_hi - pad:
        out[-1] = y_hi - pad
        for j in range(n - 2, -1, -1):
            out[j] = min(out[j], out[j + 1] - min_sep)
        for j in range(1, n):
            out[j] = max(out[j], out[j - 1] + min_sep)
    if out[0] < y_lo + pad:
        out[0] = y_lo + pad
        for j in range(1, n):
            out[j] = max(out[j], out[j - 1] + min_sep)
    resolved = [0.0] * n
    for j, idx in enumerate(order):
        resolved[idx] = float(out[j])
    return resolved


def _modality_prefix(modality: str, code: str = "") -> str:
    """Single-letter modality tag for compact force labels."""
    m = str(modality or "").strip().lower()
    c = str(code or "")
    if not m:
        if c.startswith("RXCUI|") or c.startswith("MED_"):
            m = "med"
        elif c.startswith("CCS|") or c.startswith("COND_"):
            m = "condition"
        elif c.startswith("LAB|"):
            m = "lab"
        elif c.startswith("PROC") or "procedure" in c.lower():
            m = "procedure"
    key = {
        "med": "M",
        "medication": "M",
        "condition": "C",
        "cond": "C",
        "lab": "L",
        "procedure": "P",
        "proc": "P",
    }.get(m, "")
    return key


KEY_LABEL_FONT = 5.6
KEY_LABEL_MAX_CHARS = 11


def _label_bbox_data(
    ax,
    text: str,
    x: float,
    y: float,
    ha: str,
    va: str,
    fontsize: float,
    renderer,
):
    """Measure a callout bbox in data coordinates."""
    t = ax.text(
        x, y, text,
        ha=ha, va=va,
        fontsize=fontsize, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.10",
            facecolor="white", edgecolor="none", alpha=0.82,
        ),
    )
    bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
    t.remove()
    return bb


def _bboxes_overlap(a, b, pad: float = 0.012) -> bool:
    return (
        a.x0 < b.x1 + pad and b.x0 < a.x1 + pad
        and a.y0 < b.y1 + pad and b.y0 < a.y1 + pad
    )


def _place_key_hour_labels_global(
    ax,
    jobs: list[dict],
    bar_w: float,
    nm,
    fontsize: float = KEY_LABEL_FONT,
) -> None:
    """Place all threshold-crossing labels with global bbox de-overlap."""
    if not jobs:
        return

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    th = _est_label_height(fontsize)
    min_sep = th + 0.010
    y_lo, y_hi = RISK_XLIM
    placed: list = []

    for t in ax.texts:
        txt = t.get_text()
        if not txt or "\u00b7" not in txt:
            continue
        placed.append(t.get_window_extent(renderer).transformed(ax.transData.inverted()))

    by_col: dict[float, list[dict]] = {}
    for job in jobs:
        by_col.setdefault(float(job["x_col"]), []).append(job)
    for col_jobs in by_col.values():
        col_jobs.sort(key=lambda j: float(j["mid"]))
        ideals = [float(j["mid"]) for j in col_jobs]
        ys = _resolve_vertical_label_positions(ideals, min_sep, y_lo, y_hi)
        for job, y in zip(col_jobs, ys):
            job["y_pref"] = float(y)

    jobs = sorted(
        jobs,
        key=lambda j: (-abs(float(j["w"])), float(j["x_col"]), float(j["mid"])),
    )
    dx_lanes = [0.10, 0.58, 1.06, 1.58]
    y_pad = 0.02

    for job in jobs:
        max_chars = int(job.get("max_chars", KEY_LABEL_MAX_CHARS))
        text = _label_text_compact(
            nm(job["code"], None), job["code"],
            modality=str(job.get("modality") or ""),
            max_chars=max_chars,
        )
        side = "pos" if job["pos"] else "neg"
        color = RED if job["pos"] else BLUE
        x_col = float(job["x_col"])
        x_edge = x_col + 0.5 * bar_w if side == "pos" else x_col - 0.5 * bar_w
        ha = "left" if side == "pos" else "right"
        y_anchor = float(job.get("y_anchor", job["mid"]))
        y_pref = float(job.get("y_pref", y_anchor))
        pin_y = bool(job.get("pin_y", False))
        y_opts = [y_pref]
        if not pin_y:
            for step in range(1, 12):
                y_opts.extend([y_pref + step * min_sep, y_pref - step * min_sep])
        else:
            # Keep pinned labels near the requested bar fraction; allow tiny nudges only.
            for step in range(1, 4):
                y_opts.extend([y_pref + step * 0.012, y_pref - step * 0.012])

        best = None
        best_score = 1e9
        for lane in dx_lanes:
            x_text = x_edge + lane if side == "pos" else x_edge - lane
            for y_try in y_opts:
                y_try = float(np.clip(y_try, y_lo + y_pad, y_hi - y_pad))
                bb = _label_bbox_data(ax, text, x_text, y_try, ha, "center", fontsize, renderer)
                if any(_bboxes_overlap(bb, other) for other in placed):
                    continue
                score = abs(y_try - y_anchor) + lane * 0.07
                if pin_y:
                    score += abs(y_try - y_pref) * 2.0
                if score < best_score:
                    best_score = score
                    best = (x_text, y_try, bb)
        if best is None:
            x_text = x_edge + dx_lanes[-1] if side == "pos" else x_edge - dx_lanes[-1]
            y_try = float(np.clip(y_pref, y_lo + y_pad, y_hi - y_pad))
            bb = _label_bbox_data(ax, text, x_text, y_try, ha, "center", fontsize, renderer)
            best = (x_text, y_try, bb)

        x_text, y_text, bb = best
        placed.append(bb)
        ax.plot(
            [x_edge, x_text],
            [y_anchor, y_text],
            color=color, lw=0.55, alpha=0.45, zorder=5,
            solid_capstyle="round",
        )
        ax.text(
            x_text, y_text, text,
            ha=ha, va="center",
            fontsize=fontsize, color=color, fontweight="bold",
            clip_on=False, zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.10",
                facecolor="white", edgecolor="none", alpha=0.82,
            ),
        )


def _label_text_compact(
    name: str, code: str, modality: str = "", max_chars: int = 16,
) -> str:
    """Short callout: ``M · name`` / ``C · name`` (no raw code)."""
    name = " ".join(str(name).split())
    code = str(code)
    if name == code or str(name).upper() == "PROPRIETARY":
        s = code.split("|")[-1] if "|" in code else code
    else:
        s = name
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        s = s[: max_chars - 1] + "\u2026"
    pref = _modality_prefix(modality, code)
    return f"{pref} \u00b7 {s}" if pref else s


def _place_force_labels(
    ax, cands: list[dict], x: float, bar_w: float, nm,
    fontsize: float = 6.0,
) -> float:
    """Place tip-anchored force-bar callouts (non-key columns)."""
    if not cands:
        return -np.inf

    ranked = sorted(cands, key=lambda s: abs(float(s["w"])), reverse=True)[:3]
    items = []
    for s in ranked:
        text = _label_text_compact(
            nm(s["code"], None), s["code"], modality=str(s.get("modality") or ""),
        )
        th = _est_label_height(fontsize)
        tip = float(s["hi"]) if s["pos"] else float(s["lo"])
        items.append({
            "text": text,
            "th": th,
            "pos": bool(s["pos"]),
            "color": RED if s["pos"] else BLUE,
            "tip": tip,
            "y_text": tip + (0.022 if s["pos"] else -0.022),
        })

    placed_y: list[float] = []
    for e in items:
        direction = 1.0 if e["pos"] else -1.0
        y_t = e["y_text"]
        for _ in range(10):
            need = e["th"] + 0.008
            if not any(abs(y_t - py) < need for py in placed_y):
                break
            y_t += direction * 0.036
        e["y_text"] = float(np.clip(y_t, RISK_XLIM[0] + 0.02, RISK_XLIM[1] - 0.02))
        placed_y.append(e["y_text"])

    for e in items:
        y_tip, y_text = e["tip"], e["y_text"]
        ax.plot(
            [x, x],
            [y_tip, y_text],
            color=e["color"], lw=0.55, alpha=0.45, zorder=5,
            solid_capstyle="round",
        )
        ax.text(
            x, y_text, e["text"],
            ha="center",
            va="bottom" if e["pos"] else "top",
            fontsize=fontsize, color=e["color"], fontweight="bold",
            clip_on=False, zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white", edgecolor="none", alpha=0.78,
            ),
        )
    return float(x + 0.5 * bar_w)


# Force-bar geometry with x=time, y=risk.
# Tip labels sit in the column corridor, so pitch only needs a modest gap.
FORCE_BAR_W = 0.55
FORCE_COL_GAP = 0.55
FORCE_ROW_PITCH = FORCE_BAR_W + FORCE_COL_GAP
FORCE_INCH_PER_X = 0.315
FORCE_PANEL_HEIGHT_IN = 7.2
FORCE_LABEL_PAD = 0.10
FORCE_LABEL_LEVEL_DY = 0.36
FORCE_LABEL_BAND = FORCE_COL_GAP


def _force_panel_width_in(n_batches: int) -> float:
    """Axes width in inches for one force panel with ``n_batches`` time cols."""
    n_cols = max(int(n_batches), 1) + 1  # + baseline
    data_span = n_cols * FORCE_ROW_PITCH + FORCE_BAR_W + 1.05
    return float(np.clip(data_span * FORCE_INCH_PER_X, 8.0, 28.0))


def _force_panel_height_in(n_batches: int | None = None) -> float:
    """Fixed panel height (risk axis is shared / bounded)."""
    del n_batches
    return float(FORCE_PANEL_HEIGHT_IN)


def _add_risk_trajectory_inset(
    ax,
    batches: pd.DataFrame,
    baseline: float,
    pref: str,
    pid: int | None = None,
) -> None:
    """Small risk sparkline; axes match main panel (x=time, y=risk)."""
    if batches is None or len(batches) == 0:
        return
    # AKI+ rises (early low risk) → inset in lower-right empty band;
    # AKI− falls → keep upper-right empty high-risk corner.
    if str(pref).startswith("AKI+"):
        bounds = [0.58, 0.04, 0.34, 0.28]
    else:
        bounds = [0.58, 0.68, 0.34, 0.28]

    iax = ax.inset_axes(bounds, zorder=12)
    iax.set_facecolor((1, 1, 1, 0.92))
    for sp in iax.spines.values():
        sp.set_color("0.55")
        sp.set_linewidth(0.7)

    ts = [0.0] + [float(r.time_hour) for _, r in batches.iterrows()]
    ps = [float(baseline)] + [float(r.prob) for _, r in batches.iterrows()]

    def _risk_color(r: float) -> str:
        return RED if r >= RISK_THR else BLUE

    def _draw_seg(x0: float, y0: float, x1: float, y1: float) -> None:
        # Split vertical segments that cross the threshold (y = risk).
        if abs(x1 - x0) < 1e-12 and (y0 - RISK_THR) * (y1 - RISK_THR) < 0:
            _draw_seg(x0, y0, x0, RISK_THR)
            _draw_seg(x0, RISK_THR, x1, y1)
            return
        c = _risk_color(0.5 * (y0 + y1))
        iax.plot(
            [x0, x1], [y0, y1],
            color=c, lw=1.35, solid_capstyle="butt", zorder=3,
        )

    # Staircase: horizontal (time) then vertical (risk).
    x_cur, y_cur = ts[0], ps[0]
    for i in range(1, len(ts)):
        _draw_seg(x_cur, y_cur, ts[i], ps[i - 1])
        _draw_seg(ts[i], ps[i - 1], ts[i], ps[i])
        x_cur, y_cur = ts[i], ps[i]

    for p, t in zip(ps, ts):
        iax.plot(
            [t], [p],
            marker="o", markersize=2.4, color=_risk_color(p),
            markeredgewidth=0, linestyle="none", zorder=4,
        )
    iax.axhspan(-0.02, RISK_THR, facecolor=BLUE, alpha=0.07, zorder=0, lw=0)
    iax.axhspan(RISK_THR, 1.20, facecolor=RED, alpha=0.07, zorder=0, lw=0)
    iax.axhline(
        RISK_THR, color="#4a5568", lw=0.85,
        linestyle=(0, (2.5, 1.8)), alpha=0.85, zorder=2,
    )

    n_pts = len(ps)
    imax = int(np.argmax(ps))
    imin = int(np.argmin(ps))
    t1 = max(ts) if ts else 1.0
    keyed: list[tuple[str, int]] = [("start", 0), ("end", n_pts - 1)]
    if imax not in (0, n_pts - 1) and ps[imax] >= max(ps[0], ps[-1]) + 0.05:
        if abs(ts[imax] - ts[0]) > max(0.10 * t1, 1.0):
            keyed.append(("max", imax))
    if imin not in (0, n_pts - 1) and imin != imax and ps[imin] <= min(ps[0], ps[-1]) - 0.05:
        if abs(ps[imin] - ps[-1]) >= 0.05 or ts[imin] < 0.80 * t1:
            keyed.append(("min", imin))

    declining = ps[-1] < ps[0]
    if declining:
        pref_off = {
            "start": (0.08, 0.08, "left", "bottom"),
            "max": (0.10, -0.08, "left", "top"),
            "end": (-0.06, -0.06, "right", "top"),
            "min": (-0.08, 0.08, "right", "bottom"),
        }
    else:
        pref_off = {
            "start": (0.08, -0.06, "left", "top"),
            "min": (0.10, 0.08, "left", "bottom"),
            # AKI+ end: sit directly above the terminal point.
            "end": (0.0, 0.14, "center", "bottom"),
            "max": (-0.08, -0.06, "right", "top"),
        }

    placed_xy: list[tuple[float, float]] = []
    for kind, idx in keyed:
        p, t = ps[idx], ts[idx]
        dx_f, dy, ha, va = pref_off[kind]
        tx, ty = t + dx_f * t1, p + dy
        if kind == "end" and not declining:
            # Keep the AKI+ endpoint label centered above the point, a bit higher.
            tx = float(t)
            ty = float(min(p + 0.14, RISK_XLIM[1] - 0.03))
            ha, va = "center", "bottom"
        else:
            if t <= 0.08 * t1:
                tx = max(tx, t + 0.08 * t1)
                ha = "left"
            if t >= 0.92 * t1:
                tx = min(tx, t - 0.04 * t1)
                ha = "right"
            tx = float(np.clip(tx, 0.04 * t1, 0.96 * t1))
            ty = float(np.clip(ty, 0.04, 0.96))
            for _ in range(3):
                clash = any(
                    abs(tx - px) < 0.12 * t1 and abs(ty - py) < 0.10
                    for px, py in placed_xy
                )
                if not clash:
                    break
                ty = float(np.clip(ty + 0.10, 0.04, 0.96))
        placed_xy.append((tx, ty))
        iax.annotate(
            f"{p:.2f}",
            xy=(t, p),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha, va=va,
            fontsize=5.6, fontweight="bold", color=_risk_color(p),
            arrowprops=dict(
                arrowstyle="-", color=_risk_color(p),
                lw=0.55, shrinkA=0, shrinkB=1.5,
            ),
            zorder=5, clip_on=False,
        )

    iax.set_xlim(-0.02 * max(t1, 1.0), t1 * 1.05)
    iax.set_ylim(*RISK_XLIM)
    iax.set_yticks([0.0, 0.5, 1.0])
    iax.tick_params(axis="both", labelsize=5.8, length=2.0, pad=1.2)
    iax.set_xlabel("Postoperative time (h)", fontsize=5.6, labelpad=1.0)
    iax.set_ylabel("Predicted PO-AKI probability", fontsize=5.4, labelpad=1.0)
    iax.set_title("Probability trajectory", fontsize=6.8, pad=2.0, color="0.25", loc="left")
    iax.grid(axis="y", alpha=0.22, lw=0.5)


def _draw_force_timeline_panel(
    ax, d: pd.DataFrame, batches: pd.DataFrame, nm,
    pref: str | None = None, pid: int | None = None,
    panel: str | None = None,
    tail: dict | None = None,
    inset_batches: pd.DataFrame | None = None,
) -> None:
    """Vertical force bars vs clock time (categorical x); y = risk.

    ``tail`` (optional) appends one net column after an x-axis break for the
    remaining monitoring window: keys ``time_hour``, ``prob``, ``n_events``,
    ``from_h``. ``inset_batches`` overrides the sparkline path (e.g. full 30 d).

    - Emphasized cols (risk crosses 0.5): saturated colors; else faded.
    - Rectangles abutting prior risk (neg below / pos above when mixed).
    - Labels to the right; bottom-most is closest, others only further right.
    - Emphasized cols: label every event; others: |Δrisk| > 0.05.
    """
    from matplotlib.patches import Patch, Rectangle

    batches = batches.reset_index(drop=True)
    n = len(batches)
    if n == 0:
        return

    MIN_ABS_CONTRIB = 0.05
    MAX_LABELS = 2
    KEY_MIN_CONTRIB = 0.05
    ROW_PITCH = FORCE_ROW_PITCH
    bar_w = FORCE_BAR_W

    xticks, xlabels = [], []
    cross_xs: set[float] = set()
    key_label_jobs: list[dict] = []
    pred_xs: list[float] = []
    pred_ys: list[float] = []

    # 0h baseline
    baseline = float(batches.iloc[0].prev_prob)
    x0 = 0.0
    xticks.append(x0)
    xlabels.append("0.0 h")
    pred_xs.append(x0)
    pred_ys.append(baseline)
    base_c = RED if baseline >= RISK_THR else BLUE
    ax.plot(
        [x0], [baseline],
        marker="o", color=base_c, markersize=6.0,
        markeredgecolor="white", markeredgewidth=0.7, zorder=6,
    )
    ax.plot(
        [x0, x0], [baseline - 0.012, baseline + 0.012],
        color=base_c, lw=1.4, solid_capstyle="butt", zorder=4, alpha=0.75,
    )

    for i, b in batches.iterrows():
        x = float((i + 1) * ROW_PITCH)
        xticks.append(x)
        g = d[np.isclose(d.t_key, b.t_key)].sort_values(["time_hour", "step"])
        if g.empty:
            xlabels.append(f"{float(b.time_hour):.1f} h")
            continue

        row_p0 = float(b.prev_prob)
        row_p1 = float(b.prob)
        row_crosses = (row_p0 - RISK_THR) * (row_p1 - RISK_THR) <= 0
        if not row_crosses:
            path = np.concatenate(
                [g.prev_prob.to_numpy(dtype=float), g.prob.to_numpy(dtype=float)[-1:]]
            )
            row_crosses = bool(np.any((path[:-1] - RISK_THR) * (path[1:] - RISK_THR) <= 0))

        tlab = f"{float(b.time_hour):.1f} h"
        if row_crosses:
            cross_xs.add(x)
            xlabels.append(f"\u25b8 {tlab}")
        else:
            xlabels.append(tlab)

        p_old = float(b.prev_prob)
        p_new = float(b.prob)
        segs = _layout_force_segments(g, p_old)
        for s in segs:
            if row_crosses:
                color = RED if s["pos"] else BLUE
                alpha, z = 0.92, 4
            else:
                color = RED_FADE if s["pos"] else BLUE_FADE
                alpha, z = 0.55, 2
            ax.add_patch(
                Rectangle(
                    (x - 0.5 * bar_w, s["lo"]),
                    bar_w,
                    max(s["hi"] - s["lo"], 1e-4),
                    facecolor=color, edgecolor="white", linewidth=0.55,
                    alpha=alpha, zorder=z, clip_on=True,
                )
            )
        # White dashed separators at abutting risk edges within the column.
        if len(segs) >= 2:
            edges = sorted({round(float(s["lo"]), 6) for s in segs} | {round(float(s["hi"]), 6) for s in segs})
            outer = {round(min(s["lo"] for s in segs), 6), round(max(s["hi"] for s in segs), 6)}
            for y_cut in edges:
                if y_cut in outer:
                    continue
                ax.plot(
                    [x - 0.5 * bar_w, x + 0.5 * bar_w],
                    [y_cut, y_cut],
                    color="white", lw=0.95, linestyle=(0, (2.5, 1.8)),
                    solid_capstyle="butt", zorder=5, alpha=0.92,
                )

        pred_xs.append(x)
        pred_ys.append(p_new)
        dot_c = RED if p_new >= RISK_THR else BLUE
        ax.plot(
            [x], [p_new],
            marker="o",
            color=dot_c,
            markersize=(6.6 if row_crosses else 5.6),
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=6,
        )

        t_h = float(b.time_hour)
        # Panel-A curated overrides: full label set at 4.1 h; full-name pin at 2.0 h.
        force_all_labels = (
            pid == LEGACY_POS and np.isclose(t_h, 4.1, atol=0.05)
        )
        pin_quarter_full = (
            pid == LEGACY_POS and np.isclose(t_h, 2.0, atol=0.05)
        )

        if row_crosses or force_all_labels:
            for s in segs:
                if (not force_all_labels) and float(s["w"]) <= KEY_MIN_CONTRIB:
                    continue
                job = {
                    "x_col": x,
                    "mid": float(s["mid"]),
                    "w": float(s["w"]),
                    "pos": bool(s["pos"]),
                    "code": s["code"],
                    "modality": str(s.get("modality") or ""),
                }
                if pin_quarter_full:
                    lo, hi = float(s["lo"]), float(s["hi"])
                    y_q = lo + 0.25 * (hi - lo)
                    job["y_anchor"] = y_q
                    job["mid"] = y_q  # pack / prefer around bar 1/4
                    job["max_chars"] = 0  # full name
                    job["pin_y"] = True
                key_label_jobs.append(job)
        else:
            cands = sorted(
                [s for s in segs if s["w"] > MIN_ABS_CONTRIB],
                key=lambda s: abs(float(s["w"])),
                reverse=True,
            )[:MAX_LABELS]
            _place_force_labels(ax, cands, x, bar_w, nm)

    tail_x = None
    if tail is not None and pred_ys:
        n += 1
        tail_x = float(n * ROW_PITCH)
        p_old, p_new = float(pred_ys[-1]), float(tail["prob"])
        lo, hi = sorted((p_old, p_new))
        ax.add_patch(
            Rectangle(
                (tail_x - 0.5 * bar_w, lo), bar_w, max(hi - lo, 1e-4),
                facecolor=BLUE_FADE if p_new < p_old else RED_FADE,
                edgecolor="white", linewidth=0.55, alpha=0.55, zorder=2,
            )
        )
        ax.plot(
            [tail_x], [p_new], marker="o", color=RED if p_new >= RISK_THR else BLUE,
            markersize=5.6, markeredgecolor="white", markeredgewidth=0.7, zorder=6,
        )
        ax.text(
            tail_x, hi + 0.10,
            f"\u03a3 {int(tail['n_events'])} events\n"
            f"({float(tail['from_h']):.0f}\u2013{float(tail['time_hour']):.0f} h)",
            ha="center", va="bottom", fontsize=6.4, color="0.30", zorder=8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )
        brk_x = 0.5 * (pred_xs[-1] + tail_x)
        for dx in (-0.09, 0.09):
            ax.plot(
                [brk_x + dx - 0.07, brk_x + dx + 0.07], [-0.012, 0.012],
                transform=ax.get_xaxis_transform(), color="0.25", lw=1.0,
                clip_on=False, zorder=9,
            )
        xticks.append(tail_x)
        xlabels.append(f"{float(tail['time_hour']):.0f} h (day {float(tail['time_hour']) / 24:.0f})")
        pred_xs.append(tail_x)
        pred_ys.append(p_new)

    # Gray dashed staircase (time on x, risk on y)
    if len(pred_xs) >= 2:
        sx: list[float] = [pred_xs[0]]
        sy: list[float] = [pred_ys[0]]
        for i in range(len(pred_xs) - 1):
            sx.append(pred_xs[i + 1])
            sy.append(pred_ys[i])
            sx.append(pred_xs[i + 1])
            sy.append(pred_ys[i + 1])
        ax.plot(
            sx, sy,
            color="0.35", linestyle=(0, (3.5, 2.0)), linewidth=1.55,
            zorder=4, solid_capstyle="butt", solid_joinstyle="miter",
            alpha=0.88,
        )

    thr_c = "#4a5568"
    # Soft risk bands: <0.5 light blue, ≥0.5 light red.
    y_lo, y_hi = RISK_XLIM
    bar_rects = [
        r for r in ax.patches
        if isinstance(r, matplotlib.patches.Rectangle) and r.get_width() < 5
    ]
    if bar_rects:
        y_hi = max(y_hi, max(r.get_y() + r.get_height() for r in bar_rects) + 0.03)
        # Bottom stays fixed; mark bars that continue below the axis.
        cut_cols: dict[float, tuple] = {}
        for r in bar_rects:
            if r.get_y() < y_lo:
                cut_cols.setdefault(round(r.get_x() + 0.5 * r.get_width(), 3), r.get_facecolor())
        for xc, fc in cut_cols.items():
            ax.plot(
                [xc], [y_lo + 0.035], marker="v", markersize=7.0,
                markerfacecolor="white", markeredgecolor=fc[:3], markeredgewidth=1.1,
                zorder=8, clip_on=False,
            )
    ax.axhspan(
        y_lo, RISK_THR,
        facecolor=BLUE, alpha=0.06, zorder=0, lw=0,
    )
    ax.axhspan(
        RISK_THR, y_hi,
        facecolor=RED, alpha=0.06, zorder=0, lw=0,
    )
    ax.axhspan(
        RISK_THR - 0.012, RISK_THR + 0.012,
        facecolor=thr_c, alpha=0.07, zorder=0.8, lw=0,
    )
    ax.axhline(
        RISK_THR, color=thr_c, lw=1.15, linestyle=(0, (3.2, 2.0)),
        zorder=1.6, alpha=0.90,
    )
    x_right = float(n) * ROW_PITCH
    xmin = -0.5 * bar_w - 1.05
    # Modest right pad: end-risk callout sits beside the last point (no external thr label).
    xmax = x_right + 0.5 * bar_w + 1.05
    ax.text(
        xmax - 0.08, RISK_THR + 0.018, f"Decision threshold: {RISK_THR:.2f}",
        ha="right", va="bottom",
        fontsize=6.8, fontweight="bold", color=thr_c,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white", edgecolor=thr_c, linewidth=0.65,
            alpha=0.92,
        ),
        zorder=7,
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(xmin, xmax)
    _place_key_hour_labels_global(ax, key_label_jobs, bar_w, nm)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=35, ha="right")
    for tick, xv in zip(ax.get_xticklabels(), xticks):
        if float(xv) in cross_xs:
            tick.set_fontweight("bold")
            tick.set_fontsize(8.2)
            tick.set_color("0.10")
            ax.axvspan(
                float(xv) - 0.5 * bar_w - 0.18,
                float(xv) + 0.5 * bar_w + 0.12,
                facecolor="0.88", alpha=0.32, zorder=0, lw=0,
            )
        else:
            tick.set_color("0.45")
    ax.set_xlabel("Postoperative time (h)")
    ax.set_ylabel("Predicted PO-AKI probability")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    baseline = float(batches.iloc[0].prev_prob)
    final = float(tail["prob"]) if tail is not None else float(batches.iloc[-1].prob)
    if tail is not None:
        ax.text(
            0.995, RISK_THR + 0.058,
            f"Predicted probability remained < {RISK_THR:.2f} through day "
            f"{float(tail['time_hour']) / 24:.0f} "
            f"(final {'< 0.01' if final < 0.005 else f'{final:.2f}'})",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=9.2, fontstyle="italic", color="0.20", zorder=8,
        )

    # Panel A: thin red marker at PO-AKI diagnosis time (true clock hour, not clamped).
    if pid == CURATED_POS:
        meta = (_CASE_META or _load_case_clinical_meta([int(pid)])).get(int(pid), {})
        aki_h = meta.get("aki_hours")
        if aki_h is not None and np.isfinite(float(aki_h)):
            aki_h = float(aki_h)
            aki_x = None
            for i, (_, b) in enumerate(batches.iterrows()):
                if np.isclose(float(b.time_hour), aki_h, atol=0.05):
                    aki_x = float((i + 1) * ROW_PITCH)
                    break
            if aki_x is None:
                times = [0.0] + [float(b.time_hour) for _, b in batches.iterrows()]
                xs = [0.0] + [float((i + 1) * ROW_PITCH) for i in range(len(batches))]
                t_last = float(times[-1])
                if aki_h > t_last + 1e-6 and t_last > 0:
                    # Place beyond the last event column, scaled by remaining clock time.
                    dt = max(aki_h - t_last, 1e-6)
                    span = max(t_last, 1e-6)
                    aki_x = float(xs[-1] + ROW_PITCH * min(2.0, dt / (span / max(len(batches), 1))))
                else:
                    aki_x = float(np.interp(aki_h, times, xs))
            ax.set_xlim(xmin, max(float(ax.get_xlim()[1]), aki_x + 0.55 * ROW_PITCH))
            ax.axvline(
                aki_x, color=RED, lw=1.05, linestyle=(0, (3.2, 2.0)),
                alpha=0.85, zorder=3.5, clip_on=False,
            )
            ax.text(
                aki_x - 0.12, RISK_XLIM[1] - 0.03, "PO-AKI diagnosis",
                ha="right", va="top",
                fontsize=7.0, fontweight="bold", color=RED,
                clip_on=False, zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white", edgecolor="none", alpha=0.88,
                ),
            )
    if pred_xs and pred_ys:
        ax.annotate(
            f"{baseline:.2f}",
            xy=(pred_xs[0], pred_ys[0]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color="0.20",
            zorder=8,
        )
        ax.annotate(
            f"{final:.2f}",
            xy=(pred_xs[-1], pred_ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=7.2, fontweight="bold", color="0.20",
            zorder=8,
        )
    if pref is not None and pid is not None:
        _apply_case_title(ax, pref, int(pid), baseline, final, panel=panel)
    else:
        ax.set_title(
            "Per-time force bars (prior\u2192new probability)",
            fontsize=10.2, fontweight="bold", color="0.15", loc="left", pad=12.0,
        )
    ax.legend(
        handles=[
            Patch(facecolor=RED, edgecolor="white", label="Positive arrival update"),
            Patch(facecolor=BLUE, edgecolor="white", label="Negative arrival update"),
        ],
        loc="upper left" if baseline < RISK_THR else "lower left",
        fontsize=7.2,
        frameon=False,
        title="M\u00b7med   C\u00b7condition   L\u00b7lab",
        title_fontsize=6.4,
    )
    if pref is not None:
        _add_risk_trajectory_inset(
            ax, inset_batches if inset_batches is not None else batches,
            baseline, pref, pid=pid,
        )


def _draw_case_force_only(fig, d: pd.DataFrame, nm, pref: str, pid: int) -> None:
    """Single panel: horizontal force bars vs time (with 0h baseline)."""
    batches = _batch_ends(d)
    ax = fig.add_subplot(1, 1, 1)
    _draw_force_timeline_panel(ax, d, batches, nm, pref=pref, pid=pid)


def _case_tail(raw: pd.DataFrame, d: pd.DataFrame, pid: int) -> tuple[dict | None, pd.DataFrame | None]:
    """AKI− panel: net column for arrivals after the displayed window + full-path batches."""
    if int(pid) != CURATED_NEG:
        return None, None
    full = _prepare_case_frame(raw, through_h=float(raw.time_hour.max()))
    full_b = _batch_ends(full)
    rest = full[full.time_hour > float(d.time_hour.max()) + 1e-9]
    if rest.empty:
        return None, full_b
    tail = {
        "time_hour": float(full.time_hour.max()),
        "prob": float(full_b.iloc[-1].prob),
        "n_events": int(len(rest)),
        "from_h": float(NEG_MONITOR_END_H),
    }
    return tail, full_b


def plot_cases(nm, early: pd.DataFrame, aki_pos: int, aki_neg: int) -> Path:
    _load_case_clinical_meta([int(aki_pos), int(aki_neg)])
    panels: list[tuple[int, str, str, pd.DataFrame, pd.DataFrame]] = []
    for pid, pref, letter in (
        (aki_pos, "AKI+", "A"),
        (aki_neg, "AKI\u2212", "B"),
    ):
        raw = early[early.person_id == pid]
        if raw.empty:
            raise RuntimeError(f"no early steps for person_id={pid}")
        d = _prepare_case_frame(raw, through_h=_case_through_h(int(pid)))
        batches = _batch_ends(d)
        tail, inset_b = _case_tail(raw, d, int(pid))
        panels.append((pid, pref, letter, d, batches, tail, inset_b))

    fig_w = max(
        _force_panel_width_in(len(p[4]) + (1 if p[5] else 0)) for p in panels
    ) + 0.9
    h_each = _force_panel_height_in()
    fig_h = float(2.0 * h_each + 1.70)
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=0.22,
        top=0.90,
        bottom=0.06,
        left=0.07,
        right=0.985,
    )
    for i, (pid, pref, letter, d, batches, tail, inset_b) in enumerate(panels):
        ax = fig.add_subplot(outer[i])
        _draw_force_timeline_panel(
            ax, d, batches, nm, pref=pref, pid=pid, panel=letter,
            tail=tail, inset_batches=inset_b,
        )
    # Keep the figure title clearly above panel-A's two-line subtitle band.
    fig_style.add_suptitle(
        fig, "Event-arrival attribution in representative postoperative episodes", y=0.975,
    )
    out = FIG / "arrival_case_studies_main.png"
    fig.savefig(out, dpi=180)
    plt.close()
    return out


def plot_case_with_stems(nm, early: pd.DataFrame, pid: int, tag: str) -> Path:
    """Hero case: force bars only (x=time, y=risk), with 0h baseline."""
    _load_case_clinical_meta([int(pid)])
    raw = early[early.person_id == pid]
    if raw.empty:
        raise RuntimeError(f"no early steps for person_id={pid}")
    d = _prepare_case_frame(raw, through_h=_case_through_h(int(pid)))
    batches = _batch_ends(d)
    tail, inset_b = _case_tail(raw, d, int(pid))
    fig_w = _force_panel_width_in(len(batches) + (1 if tail else 0)) + 0.8
    fig_h = _force_panel_height_in() + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(1, 1, 1)
    _draw_force_timeline_panel(
        ax, d, batches, nm, pref=tag, pid=pid, tail=tail, inset_batches=inset_b,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.12)
    out = FIG / ("arrival_case_stem_aki_pos.png" if tag.startswith("AKI+") else "arrival_case_stem_aki_neg.png")
    fig.savefig(out, dpi=180)
    alt = FIG / (
        "arrival_case_waterfall_aki_pos.png" if tag.startswith("AKI+") else "arrival_case_waterfall_aki_neg.png"
    )
    fig.savefig(alt, dpi=180)
    plt.close()
    return out


def plot_case_figures_only(
    steps_path: Path,
    aki_pos: int,
    aki_neg: int,
) -> None:
    """Regenerate force-bar case figures from an order-robust steps CSV."""
    FIG.mkdir(parents=True, exist_ok=True)
    nm = _names()
    steps = pd.read_csv(steps_path)
    steps["person_id"] = steps.person_id.astype(int)
    # Pass raw steps only — plot_cases / plot_case_with_stems crop once.
    # (Double _prepare_case_frame previously truncated panel A to ~5.5 h and
    # clamped the 8.0 h diagnosis marker to the last column.)
    early = steps[
        steps.person_id.isin([int(aki_pos), int(aki_neg)])
    ].copy()
    p_cases = plot_cases(nm, early, int(aki_pos), int(aki_neg))
    p_stem_pos = plot_case_with_stems(nm, early, int(aki_pos), "AKI+")
    p_stem_neg = plot_case_with_stems(nm, early, int(aki_neg), "AKI\u2212")
    print("wrote", p_cases)
    print("wrote", p_stem_pos)
    print("wrote", p_stem_neg)
    (FIG / "case_ids.json").write_text(
        json.dumps(
            {
                "aki_pos": aki_pos,
                "aki_neg": aki_neg,
                "horizon_h": HORIZON_H,
                "neg_monitor_end_h": NEG_MONITOR_END_H,
                "attribution": "order-robust hybrid (Shapley-MC within exact time_hour ties)",
                "steps": str(steps_path),
            },
            indent=2,
        )
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("cases", "ranking", "all"),
        default="cases",
        help="cases: force-bar case studies; ranking: pooled code ranks; all: both",
    )
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    nm = _names()

    case_meta = FIG / "case_ids.json"
    do_cases = args.mode in ("cases", "all") and CASE_STEPS.exists() and case_meta.exists()
    do_ranking = args.mode in ("ranking", "all")

    if do_cases:
        meta = json.loads(case_meta.read_text())
        print(f"using order-robust case steps: {CASE_STEPS}")
        plot_case_figures_only(
            CASE_STEPS, int(meta["aki_pos"]), int(meta["aki_neg"]),
        )
        print("done case figures", FIG)
        if args.mode == "cases":
            return

    if not do_ranking:
        return

    rep = pd.read_csv(POOL / "event_arrival_by_code_replicated.csv")
    rep = _add_fold_ci(rep)
    pooled = pd.read_csv(POOL / "event_arrival_by_code_pooled.csv")
    pooled = _add_fold_ci(pooled)

    p_rank = plot_symbolic_rank(nm, rep)

    print("pooling postoperative 7d arrival codes from fold steps...")
    pooled_7d = _pool_arrival_code_table(horizon_h=POSTOP_7D_H)
    POOL.mkdir(parents=True, exist_ok=True)
    pooled_7d.to_csv(POOL / "event_arrival_by_code_7d_pooled.csv", index=False)
    rep_7d = pooled_7d[pooled_7d.folds_agreeing_sign == 1].copy()
    rep_7d.to_csv(POOL / "event_arrival_by_code_7d_replicated.csv", index=False)
    p_rank_7d = plot_symbolic_rank(
        nm,
        rep_7d,
        out_name="arrival_rank_symbolic_7d.png",
        suptitle="Code-level event-arrival attribution during the first 7 postoperative days",
        xlab="Mean arrival \u0394logit",
        left_title="Largest upward risk updates",
        right_title="Largest downward risk updates",
        panel_letters=("A", "B"),
    )
    p_conc = plot_concordance(nm, rep)
    print("wrote", p_rank)
    print("wrote", p_rank_7d)
    print("wrote", p_conc)

    meta = json.loads(case_meta.read_text()) if case_meta.exists() else {}
    aki_pos, aki_neg = meta.get("aki_pos"), meta.get("aki_neg")

    summary = {
        "n_pooled_codes": int(len(pooled)),
        "n_replicated": int(len(rep)),
        "fold_summaries": {
            f: json.loads(
                (ROOT / f"result/event_stream/transformer/event_arrival_fold{f}/event_arrival_summary.json").read_text()
            )
            for f in range(1, 6)
        },
        "cases": {"aki_pos": aki_pos, "aki_neg": aki_neg},
        "figures": [p.name for p in [p_rank, p_rank_7d, p_conc]],
    }
    (FIG / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done", FIG)


if __name__ == "__main__":
    main()
