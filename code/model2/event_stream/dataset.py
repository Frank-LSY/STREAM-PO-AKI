from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from event_stream.input_builder import episode_to_sequence_features


def load_episode_jsonl(path: str) -> List[Dict[str, Any]]:
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_episode_jsonl_many(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        out.extend(load_episode_jsonl(path))
    return out


class PeriopEpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        code_vocab: Dict[str, int],
        lab_stats: Optional[Dict[str, Any]] = None,
        age_stats: Optional[Dict[str, Any]] = None,
        num_med_exposure: int = 0,
        *,
        baseline_token: str = "dedicated",
        drop_conditions: bool = False,
        drop_lab_values: bool = False,
        drop_labs: bool = False,
        drop_meds: bool = False,
        generic_surgery: bool = False,
    ):
        self.episodes = episodes
        self.code_vocab = code_vocab
        self.lab_stats = lab_stats
        self.age_stats = age_stats
        self.num_med_exposure = int(num_med_exposure)
        self.baseline_token = str(baseline_token)
        self.drop_conditions = bool(drop_conditions)
        self.drop_lab_values = bool(drop_lab_values)
        self.drop_labs = bool(drop_labs)
        self.drop_meds = bool(drop_meds)
        self.generic_surgery = bool(generic_surgery)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return episode_to_sequence_features(
            self.episodes[idx],
            self.code_vocab,
            add_base=True,
            lab_stats=self.lab_stats,
            age_stats=self.age_stats,
            num_med_exposure=self.num_med_exposure,
            baseline_token=self.baseline_token,
            drop_conditions=self.drop_conditions,
            drop_lab_values=self.drop_lab_values,
            drop_labs=self.drop_labs,
            drop_meds=self.drop_meds,
            generic_surgery=self.generic_surgery,
        )


def _pad_1d(seq_vals, pad_len, pad_value):
    return list(seq_vals) + [pad_value] * pad_len


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["sequence"]) for x in batch)

    med_dim = 0
    for item in batch:
        for x in item["sequence"]:
            m = x.get("med_day_feats")
            if m is not None:
                med_dim = max(med_dim, len(m))

    rows: Dict[str, list] = {
        "code_ids": [],
        "modality_ids": [],
        "event_type_ids": [],
        "time_days": [],
        "time_hours": [],
        "values": [],
        "value_masks": [],
        "age_values": [],
        "age_masks": [],
        "attention_mask": [],
        "base_mask": [],
        "preop_mask": [],
        "postop_mask": [],
        "procedure_mask": [],
        "mem_candidate_mask": [],
        "med_exposure_indices": [],
        "labels": [],
    }
    med_day_feats: List[List[List[float]]] = []

    for item in batch:
        seq = item["sequence"]
        pad_len = max_len - len(seq)
        rows["code_ids"].append(_pad_1d([x["code_id"] for x in seq], pad_len, 0))
        rows["modality_ids"].append(_pad_1d([x["modality_id"] for x in seq], pad_len, 0))
        rows["event_type_ids"].append(_pad_1d([x.get("event_type_id", 0) for x in seq], pad_len, 0))
        rows["time_days"].append(_pad_1d([x["time_day"] for x in seq], pad_len, 0.0))
        rows["time_hours"].append(_pad_1d([x["time_hour"] for x in seq], pad_len, 0.0))
        rows["values"].append(_pad_1d([x["value"] for x in seq], pad_len, 0.0))
        rows["value_masks"].append(_pad_1d([x["value_mask"] for x in seq], pad_len, 0.0))
        rows["age_values"].append(_pad_1d([x.get("age_value", 0.0) for x in seq], pad_len, 0.0))
        rows["age_masks"].append(_pad_1d([x.get("age_mask", 0.0) for x in seq], pad_len, 0.0))
        rows["attention_mask"].append([1.0] * len(seq) + [0.0] * pad_len)
        rows["base_mask"].append(_pad_1d([x.get("is_base", 0) for x in seq], pad_len, 0))
        rows["preop_mask"].append(_pad_1d([x.get("is_preop", 0) for x in seq], pad_len, 0))
        rows["postop_mask"].append(_pad_1d([x.get("is_postop", 0) for x in seq], pad_len, 0))
        rows["procedure_mask"].append(_pad_1d([x.get("is_procedure", 0) for x in seq], pad_len, 0))
        rows["mem_candidate_mask"].append(_pad_1d([x.get("mem_candidate", 0) for x in seq], pad_len, 0))
        rows["med_exposure_indices"].append(_pad_1d([x.get("med_exposure_idx", -1) for x in seq], pad_len, -1))
        rows["labels"].append(item["label"])

        if med_dim > 0:
            feat_rows = []
            for x in seq:
                r = x.get("med_day_feats")
                if not r:
                    r = [0.0] * med_dim
                elif len(r) < med_dim:
                    r = list(r) + [0.0] * (med_dim - len(r))
                else:
                    r = list(r[:med_dim])
                feat_rows.append(r)
            feat_rows.extend([[0.0] * med_dim for _ in range(pad_len)])
            med_day_feats.append(feat_rows)

    out: Dict[str, torch.Tensor] = {
        "code_ids": torch.tensor(rows["code_ids"], dtype=torch.long),
        "modality_ids": torch.tensor(rows["modality_ids"], dtype=torch.long),
        "event_type_ids": torch.tensor(rows["event_type_ids"], dtype=torch.long),
        "time_days": torch.tensor(rows["time_days"], dtype=torch.float),
        "time_hours": torch.tensor(rows["time_hours"], dtype=torch.float),
        "values": torch.tensor(rows["values"], dtype=torch.float),
        "value_masks": torch.tensor(rows["value_masks"], dtype=torch.float),
        "age_values": torch.tensor(rows["age_values"], dtype=torch.float),
        "age_masks": torch.tensor(rows["age_masks"], dtype=torch.float),
        "attention_mask": torch.tensor(rows["attention_mask"], dtype=torch.float),
        "base_mask": torch.tensor(rows["base_mask"], dtype=torch.float),
        "preop_mask": torch.tensor(rows["preop_mask"], dtype=torch.float),
        "postop_mask": torch.tensor(rows["postop_mask"], dtype=torch.float),
        "procedure_mask": torch.tensor(rows["procedure_mask"], dtype=torch.float),
        "mem_candidate_mask": torch.tensor(rows["mem_candidate_mask"], dtype=torch.float),
        "med_exposure_indices": torch.tensor(rows["med_exposure_indices"], dtype=torch.long),
        "labels": torch.tensor(rows["labels"], dtype=torch.float),
    }
    if med_dim > 0:
        out["med_day_feats"] = torch.tensor(med_day_feats, dtype=torch.float)
    return out
