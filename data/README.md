# data/

This folder ships only **reference vocabularies**. Patient-level files are created here locally by the data-preparation scripts and are excluded by `.gitignore`.

## Included reference files

| File | Content |
|---|---|
| `ccs_codes.csv`, `ccs_codes.json` | HCUP single-level CCS diagnosis categories (code → name; list of categories used) |
| `snomed_procedure_names.json` | SNOMED CT names of the index procedure types |
| `procedure_region_map.json` | Index procedure → anatomical region (abdomen, joint, spine, …) |
| `rxcui_to_name.csv`, `top90_rxcuis.json` | RxNorm ingredient names; ingredient list used as medication tokens |
| `lab_codes.json`, `lab_concept_names.json` | OMOP measurement concepts used as laboratory tokens, with names |

`code/icd_ccs.py` also needs the HCUP ICD-9-CM and ICD-10-CM → CCS crosswalks. Download them from AHRQ HCUP and save them as `reference/ccs_icd9cm.csv` and `reference/ccs_icd10cm.csv`, or point `CCS_MAP_DIR` at their location.

## Files created locally (not distributed)

```
data/
├── five_folds_deduplicate4/   # episode JSONL: 7d_window_clm.mixed.fold{1..5}.jsonl, 7d_window_clm.mixed.test.jsonl
├── five_folds_xgboost4/       # static tabular features (logistic regression, XGBoost)
└── five_folds_ts6h4/          # 6-h grid arrays (train/val/test .npz, GRU-D .obs.npy masks, vocab/)
```

## Episode JSONL format

One surgical episode per line. Times are stored relative to the index surgery (`time_day`, `time_hour`); `phase` is 0 before and 1 after surgery.

```text
{
  "person_id": str,              # site identifier (never distributed)
  "label_aki": int,              # coded PO-AKI within 30 days (0/1)
  "aki_time": str | null,
  "t0", "window_start", "window_end": str,
  "preop_days": int, "postop_days": int,
  "static_tokens":    [{"code": "AGE"|"SEX|<id>"|"RACE|<id>", "value": float|null, "time_day", "time_hour", "phase", "modality", "time"}],
  "procedure_tokens": [{"code": "SNOMED|<id>", ...}],
  "condition_events": [{"code": "CCS|<n>", "raw_code": "ICD10CM|...", ...}],
  "lab_events":       [{"code": "LAB|<concept_id>", "lab_code": int, "unit": str, "value": float, ...}],
  "med_events":       [{"code": "RXCUI|<id>", "rxcui": str, "event_type": "start"|"end", "raw_code": str, ...}],
  "events":           [...],     # all events merged in time order
  "selected_lab_names": [int], "selected_rxcui_names": [str],
  "med_day_vector": [[int]]      # day-level medication exposure (used by the time-series features)
}
```
