#!/usr/bin/env python3
"""Strip CCS|157 (PO-AKI outcome-equivalent conditions) from episode JSONL files.

The PO-AKI label is unchanged. Only predictor condition tokens whose code is CCS|157
(or whose raw ICD is 584.x / N17.x / N19) are removed from ``events`` and
``condition_events``. Use after updating ``process_data.py``, to clean already-built
episode files without a full OMOP re-extract.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ICD_AKI_RE = re.compile(r"(?:^|\|)((?:N17(?:\.\d+)?)|(?:584(?:\.\d+)?)|(?:N19(?:\.\d+)?))$", re.I)


def is_excluded_condition_event(ev: Dict[str, Any]) -> bool:
    if str(ev.get("modality", "")).lower() != "condition":
        return False
    code = str(ev.get("code", "") or "")
    if code == "CCS|157":
        return True
    raw = str(ev.get("raw_code", "") or ev.get("condition_source_value", "") or "")
    return bool(ICD_AKI_RE.search(raw.strip().upper().replace(" ", "")))


def scrub_episode(ep: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    n_drop = 0
    for key in ("events", "condition_events"):
        seq: List[Dict[str, Any]] = list(ep.get(key) or [])
        kept = []
        for ev in seq:
            if is_excluded_condition_event(ev):
                n_drop += 1
                continue
            kept.append(ev)
        ep[key] = kept
    return ep, n_drop


def scrub_file(path: Path, inplace: bool) -> Dict[str, int]:
    out_path = path if inplace else path.with_suffix(path.suffix + ".no157")
    tmp = path.with_suffix(path.suffix + ".tmp")
    n_ep = n_touch = n_drop = 0
    with path.open() as fin, tmp.open("w") as fout:
        for line in fin:
            ep = json.loads(line)
            ep, dropped = scrub_episode(ep)
            n_ep += 1
            n_drop += dropped
            if dropped:
                n_touch += 1
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return {"episodes": n_ep, "episodes_touched": n_touch, "events_dropped": n_drop}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=str, help="Episode JSONL path(s) or a directory.")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite each file in place (via temp rename).")
    args = p.parse_args()

    paths: List[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("7d_window_clm*.jsonl")))
        else:
            paths.append(path)
    # skip temps / already-scrubbed sidecars
    paths = [p for p in paths if not p.name.endswith(".tmp") and ".no157" not in p.name]

    for path in paths:
        stats = scrub_file(path, inplace=args.inplace)
        dest = path if args.inplace else path.with_suffix(path.suffix + ".no157")
        print(f"{path.name}: dropped {stats['events_dropped']} events in "
              f"{stats['episodes_touched']}/{stats['episodes']} episodes → {dest}")


if __name__ == "__main__":
    main()
