"""Recompute only the GRU-D entry of the 5-fold multimodel landmark calibration summary.

Writes a merged multimodel_summary.json (other models reused unchanged) for
``evaluate_landmark_calibration_multimodel.py --reuse_summary_json``.
"""
import argparse
import json
from pathlib import Path

from common.landmark_calibration import aggregate_fold_summaries
from event_stream.dataset import load_episode_jsonl
from event_stream.evaluate_landmark_calibration_multimodel import collect_ts_trajectories, run_model_summaries
from common.paths import PROJECT_ROOT


def slim(summ):
    out = []
    for s in summ:
        d = {k: v for k, v in s.items() if k not in ("curve", "curves_by_fold")}
        c = s.get("curve") or s.get("curve_mean") or {}
        d["curve"] = {k: c[k] for k in ("n_bins", "bin_pred", "bin_obs", "bin_n", "n_boot",
                                         "bin_obs_lo", "bin_obs_hi", "ci", "n_folds") if k in c}
        if "pred_hist" in s:
            d["pred_hist"] = s["pred_hist"]
        if "curve_mean" in s and "curve" not in s:
            d["curve_mean"] = s["curve_mean"]
        out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prev_summary", required=True)
    p.add_argument("--out_summary", required=True)
    p.add_argument("--ts_root", default=str(PROJECT_ROOT / "result/timeseries/grud"))
    p.add_argument("--ts_npz", default=str(PROJECT_ROOT / "data/five_folds_ts6h4/test.npz"))
    p.add_argument("--episode_path",
                   default=str(PROJECT_ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ts_batch_size", type=int, default=64)
    p.add_argument("--horizon_h", type=float, default=720.0)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prev = json.loads(Path(args.prev_summary).read_text())
    landmarks_h = prev["landmarks_h"]
    episodes = load_episode_jsonl(args.episode_path)
    keep_pids = [str(ep.get("person_id", "")) for ep in episodes]
    meta_by_pid = {str(ep.get("person_id", "")): {"aki_time": ep.get("aki_time"), "t0": ep.get("t0")} for ep in episodes}

    per_fold = []
    for fold in prev["folds"]:
        trajs = collect_ts_trajectories(str(Path(args.ts_root) / f"fold{fold}/best_model.pt"), Path(args.ts_npz),
                                        meta_by_pid, keep_pids, device_str=args.device, batch_size=args.ts_batch_size)
        summ, meta, _ = run_model_summaries(trajs, landmarks_h=landmarks_h, horizon_h=args.horizon_h,
                                            n_bins=args.n_bins, n_boot=args.n_boot, seed=args.seed + 17 + fold, fold=fold)
        per_fold.append(summ)
        prev["meta"][f"fold{fold}"]["ts_grud"] = meta
        print(f"fold{fold}: trajectories={len(trajs)}", flush=True)

    prev["summaries"]["ts_grud"] = slim(aggregate_fold_summaries(per_fold))
    Path(args.out_summary).write_text(json.dumps(prev))
    print("wrote", args.out_summary)


if __name__ == "__main__":
    main()
