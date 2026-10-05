from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, NUM_POSTOP_BUCKETS, SEQ_LEN


def load_npz_episodes(path: Path, mmap: bool = True) -> Dict[str, Any]:
    """Load episode arrays from an .npz archive.

    Note: NumPy cannot memmap arrays inside .npz; ``mmap_mode`` is ignored for
    archive members and ``features`` is fully materialized in RAM.
    """
    mode = "r" if mmap else None
    data = np.load(path, allow_pickle=True, mmap_mode=mode)
    return {
        "features": data["features"],
        "labels": np.asarray(data["labels"], dtype=np.float32),
        "person_ids": data["person_ids"],
        "max_postop_bucket": np.asarray(data["max_postop_bucket"], dtype=np.int64),
        "_npz": data,
    }


def load_feature_layout(data_root: Path) -> Dict[str, Any]:
    with (data_root / "vocab" / "feature_layout.json").open(encoding="utf-8") as f:
        return json.load(f)


def _collate_indices(dataset: "Ts6hEpisodeDataset", indices: Sequence[int]) -> Dict[str, Any]:
    """Vectorized batch gather: one fancy-index + one f16→f32 cast for the whole batch."""
    idx = np.asarray(indices, dtype=np.int64)
    feats = np.asarray(dataset.features[idx], dtype=np.float32)
    labels = torch.from_numpy(np.asarray(dataset.labels[idx], dtype=np.float32))
    max_postop = torch.from_numpy(np.asarray(dataset.max_postop_bucket[idx], dtype=np.int64))
    person_ids = [str(dataset.person_ids[int(i)]) for i in idx]

    B = int(idx.shape[0])
    preop = torch.zeros(B, SEQ_LEN)
    base = torch.zeros(B, SEQ_LEN)
    postop = torch.zeros(B, SEQ_LEN)
    preop[:, :BASE_BUCKET_IDX] = 1.0
    base[:, BASE_BUCKET_IDX] = 1.0
    postop[:, BASE_BUCKET_IDX + 1 :] = 1.0

    postop_valid = torch.zeros(B, SEQ_LEN)
    last = np.minimum(dataset.max_postop_bucket[idx], NUM_POSTOP_BUCKETS - 1)
    for i, last_j in enumerate(last.tolist()):
        postop_valid[i, BASE_BUCKET_IDX + 1 : BASE_BUCKET_IDX + 1 + int(last_j) + 1] = 1.0

    online_mask = postop * postop_valid

    extra: Dict[str, Any] = {}
    if dataset.obs is not None:
        obs = np.unpackbits(dataset.obs[idx], axis=-1, count=dataset.n_obs)
        extra["obs_mask"] = torch.from_numpy(obs.astype(np.float32))

    return {
        **extra,
        "bucket_features": torch.from_numpy(feats),
        "labels": labels,
        "person_ids": person_ids,
        "attention_mask": torch.ones(B, SEQ_LEN),
        "base_mask": base,
        "preop_mask": preop,
        "postop_mask": postop,
        "online_mask": online_mask,
        "max_postop_bucket": max_postop,
    }


def obs_mask_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + ".obs.npy")


class Ts6hEpisodeDataset(Dataset):
    def __init__(self, npz_path: str | Path, load_obs: bool = False):
        npz_path = Path(npz_path)
        self.data = load_npz_episodes(npz_path)
        self.features = self.data["features"]
        self.labels = self.data["labels"]
        self.person_ids = self.data["person_ids"]
        self.max_postop_bucket = self.data["max_postop_bucket"]
        self.n = int(self.labels.shape[0])
        self.obs = None
        self.n_obs = 0
        if load_obs:
            meta_path = next(d / "vocab" / "obs_mask.json" for d in npz_path.parents
                             if (d / "vocab" / "obs_mask.json").exists())
            self.n_obs = int(json.loads(meta_path.read_text())["n_obs"])
            self.obs = np.load(obs_mask_path(npz_path))
            if self.obs.shape[0] != self.n:
                raise RuntimeError(f"obs mask rows {self.obs.shape[0]} != {self.n} for {npz_path}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        # Return index only; batching is vectorized in collate_fn.
        return int(idx)

    def collate_fn(self, indices: List[int]) -> Dict[str, Any]:
        return _collate_indices(self, indices)


def collate_ts6h(batch: List[Any]) -> Dict[str, Any]:
    """Backward-compatible name; expects a dataset-bound collate via ``dataset.collate_fn``.

    If called with raw indices this will fail — always pass ``collate_fn=ds.collate_fn``.
    """
    raise TypeError(
        "Use Ts6hEpisodeDataset.collate_fn (vectorized). "
        "Example: DataLoader(ds, collate_fn=ds.collate_fn, ...)"
    )


def batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
