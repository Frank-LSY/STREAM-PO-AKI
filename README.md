# STREAM: Event-stream Transformer for dynamic postoperative AKI prediction

This repository contains the model, training, evaluation and analysis code, plus aggregate results, for **STREAM**, an event-stream Transformer that predicts postoperative acute kidney injury (PO-AKI) from perioperative EHR data.

STREAM reads each surgical episode as one irregular sequence of time-stamped clinical events: diagnoses (CCS categories), medication starts and stops (RxNorm ingredients) and laboratory measurements, from 7 days before surgery to 30 days after. A dedicated `[BASE]` token produces a risk estimate at the surgery index, and the model then updates the risk after every postoperative event. It is compared with static models (logistic regression, XGBoost) and fixed-window time-series models (LSTM, GRU-D, Time-series Transformer) that use the same cohort and information.

> **No patient data are included.** The study used electronic health records that cannot be shared. This repository holds code, public vocabulary mappings and aggregate results only. See [Data availability and privacy](#data-availability-and-privacy).

## Main results

Held-out temporal test set (23,538 surgical episodes, 10.1% with coded PO-AKI). Values are percentages, shown as the mean (min, max) of five fold-specific models. Dynamic metrics are macro-averaged over 120 six-hour postoperative windows (0–720 h).

| Model | Baseline AUROC | Baseline AUPRC | Dynamic macro-AUROC | Dynamic macro-AUPRC |
|---|---|---|---|---|
| Logistic regression | 91.05 (91.02, 91.07) | 61.17 (61.08, 61.26) | — | — |
| XGBoost | 90.69 (90.57, 90.88) | 60.47 (60.24, 60.68) | — | — |
| LSTM | 92.42 (92.21, 92.56) | 67.76 (66.61, 68.34) | 93.24 (92.79, 93.59) | 44.49 (41.97, 46.35) |
| GRU-D | 92.63 (92.51, 92.83) | 69.48 (69.04, 70.28) | 94.23 (93.98, 94.39) | 54.70 (53.37, 55.41) |
| Time-series Transformer | 93.33 (93.27, 93.41) | 72.88 (72.72, 73.07) | 94.76 (94.59, 94.86) | 52.45 (50.72, 53.85) |
| **STREAM** | **94.19** (94.02, 94.32) | **77.14** (76.70, 77.73) | **94.82** (94.72, 94.96) | **58.86** (58.06, 60.23) |

All paper tables are in [`paper_results/tables/PAPER_TABLES.md`](paper_results/tables/PAPER_TABLES.md), the manuscript figures in [`paper_results/figures/`](paper_results/figures/), and machine-readable source data in [`paper_results/source_data/`](paper_results/source_data/).

## Repository layout

```
STREAM-PO-AKI/
├── code/
│   ├── get_clm.py                       # extract 7-day-pre / 30-day-post windows from an OMOP CDM database
│   ├── icd_ccs.py                       # ICD-9/10-CM -> CCS mapping (needs HCUP crosswalks in data/reference/)
│   ├── process_data.py                  # build episode JSONL (tokens, phases, time offsets)
│   ├── build_five_folds_deduplicate4.py # temporal test split + 5-fold development split
│   ├── scrub_ccs157_from_episodes.py    # remove the AKI-defining CCS category from inputs
│   ├── build_table3_patient_characteristics.py  # Table 1
│   ├── run_descriptive_analysis_dedup4.py       # cohort descriptives
│   ├── plot_fig_cohort_landscape.py             # cohort figure
│   └── model2/
│       ├── common/          # shared utilities: metrics, encoders, figure style, paths
│       ├── event_stream/    # STREAM: model, training, evaluation, interpretability, figures
│       ├── timeseries/      # 6-h grid dataset, LSTM / GRU-D / Transformer, evaluation
│       ├── static/          # tabular dataset, logistic regression, XGBoost
│       ├── scripts/         # shell drivers (5-fold training/evaluation) and table builders
│       └── *.sh             # thin wrappers around scripts/ (see below)
├── data/                    # reference vocabularies only; patient data go here locally (gitignored)
└── paper_results/
    ├── figures/             # manuscript figures
    ├── tables/              # PAPER_TABLES.md
    └── source_data/         # aggregate CSVs behind every table and figure
```

Training and evaluation write to `result/` at the repository root (gitignored).

## Installation

Python 3.12 with a CUDA-capable GPU (results were produced on NVIDIA H100 GPUs).

```bash
conda create -n stream python=3.12 -y
conda activate stream
pip install -r requirements.txt
```

Every Python entry point under `code/model2` imports `common.*`, so set:

```bash
export PERIOP_ROOT=/path/to/STREAM-PO-AKI        # optional; defaults to the repository root
export PYTHONPATH=$PERIOP_ROOT/code/model2
export PYTHON=python                              # interpreter used by the shell drivers
```

The shell drivers set `PYTHONPATH` themselves. They read data from `data/` and write results to `result/`.

## Data preparation

The pipeline expects an OMOP CDM v5.4 database. Steps 1–3 need database access; everything after that runs on the derived files.

1. **Extract episodes** (SQL Server via ODBC). Connection settings come from environment variables:
   ```bash
   export OMOP_SERVER=... OMOP_DATABASE=... OMOP_USER=... OMOP_PASSWORD=...
   export OMOP_SCHEMA="[MyOmopDb].[v5_4]"
   cd code && python get_clm.py
   ```
   `get_clm.py` expects a cohort file `data/all_conditions_clean.csv` (one row per index procedure, with `person_id`, demographics and procedure time). It needs the HCUP CCS crosswalks `ccs_icd9cm.csv` and `ccs_icd10cm.csv` in `data/reference/`, and calls the public NLM RxNav API to map NDC codes to RxNorm ingredients.
2. **Build episodes and splits** (from `code/`):
   ```bash
   python process_data.py -i <raw_extract.jsonl> -o <episodes.jsonl>      # once for PO-AKI and once for non-PO-AKI episodes
   python build_five_folds_deduplicate4.py --pos_path <aki_episodes.jsonl> --neg_path <non_aki_episodes.jsonl>
   python scrub_ccs157_from_episodes.py ../data/five_folds_deduplicate4 --inplace   # drop the AKI-defining CCS category from inputs
   ```
   This produces `data/five_folds_deduplicate4/7d_window_clm.mixed.fold{1..5}.jsonl` (development folds) and `7d_window_clm.mixed.test.jsonl` (the temporal test set).
3. **Derived inputs for the comparison models** (from `code/model2`):
   ```bash
   bash static_build_dataset_xgboost.sh                 # data/five_folds_xgboost4/   (static tabular)
   bash timeseries_build_dataset.sh                     # data/five_folds_ts6h4/      (6-h grid, 149 steps x 548 features)
   python timeseries/build_obs_masks.py                 # GRU-D observation masks (written next to the 6-h grid arrays)
   ```

The episode format is described in [`data/README.md`](data/README.md).

## Training and evaluation

All commands run from `code/model2`. Each driver trains or evaluates the five fold models. With several idle GPUs the folds run in parallel; with one GPU they run sequentially.

| Step | Command |
|---|---|
| STREAM (event-stream Transformer), train 5 folds | `bash event_stream_train_all_folds_transformer.sh` |
| STREAM baseline (surgery-index) evaluation | `bash scripts/event_stream/eval_baseline.sh transformer` |
| STREAM dynamic evaluation on the 6-h patient-window grid | `bash scripts/event_stream/eval.sh transformer fixedgrid` |
| Static models | `bash static_train_eval_logistic.sh` and `bash static_train_eval_xgboost.sh` |
| Time-series models, train 5 folds | `TS_FORCE_SEQUENTIAL=1 bash scripts/timeseries/train_all_folds.sh {transformer,lstm,grud}` |
| Time-series baseline / dynamic evaluation | `bash timeseries_eval_baseline_grud.sh`, `bash scripts/timeseries/eval_postop6h.sh ARCH` |
| Ablations (A1–A5, I1–I4) | `bash scripts/event_stream/train_ablations.sh`, then `bash scripts/event_stream/eval_ablations.sh` and `bash scripts/event_stream/run_fixedgrid_ablation.sh` |

`TS_FORCE_SEQUENTIAL=1` trains time-series folds one at a time. The 6-h grid arrays are large, and loading several folds at once can saturate host memory and I/O.

Key STREAM settings (see `scripts/event_stream/train_all_folds_transformer.sh`): block attention (bidirectional preoperative, causal postoperative), dedicated `[BASE]` token, continuous time encoding, losses λ_online = 1.0, λ_base = 0.3, λ_MEM = 0.01 with 15% masking, AdamW at 1e-4 with step decay, early stopping on 0.9 × validation macro-AUPRC + 0.1 × baseline AUPRC.

## Reproducing the paper tables and figures

From `code/model2`, after training and evaluation:

| Output | Command |
|---|---|
| Table 2 / Table S5 (macro dynamic performance) | `python scripts/summarize_dynamic_macro.py` |
| Table 3, Tables S7–S8 (ablations) | `python scripts/event_stream/ablation_full_metrics_table.py --sd` and `python scripts/event_stream/ablation_paired_table.py` |
| Table 1 | `python ../build_table3_patient_characteristics.py` |
| Calibration figure and table | `python event_stream/evaluate_landmark_calibration_multimodel.py --folds 1,2,3,4,5 --max_episodes 0 --dpi 300` |
| Dynamic AUPRC figure | `python plot_dynamic_auprc_comparison.py` |
| Interpretability figure (age / procedure substitution, grouped deletion) | `python event_stream/plot_interpretability_main.py` |
| Event-arrival case studies and code-level ranking | `python event_stream/plot_event_arrival_main.py` and `... --mode ranking` |
| Within-window order sensitivity | `python event_stream/plot_order_sensitivity_figures.py` |
| Time-series Transformer case contrast | `python timeseries/plot_ts_case_contrast.py` |
| Cohort figure | `python ../plot_fig_cohort_landscape.py` |
| Token vocabulary (supplement) | `python scripts/event_stream/export_token_vocabulary.py` |

The interpretability figures depend on upstream analyses in `event_stream/` (`evaluate_age_doseresponse.py`, `evaluate_procedure_effects.py`, `evaluate_event_contribution_time.py`, `evaluate_event_arrival.py`, `evaluate_within_bin_order_sensitivity.py`) and their pooling scripts.

The two case-study figures show individual episodes. Their `person_id`s are not part of this repository. To regenerate them on your own data, choose cases and set `STREAM_CASE_POS_ID` and `STREAM_CASE_NEG_ID`.

## Data availability and privacy

- The EHR data come from the Memorial Hermann Health System and cannot be shared publicly under the data-use agreement and IRB approval. No patient-level data, identifiers, dates, model checkpoints or per-patient predictions are included here.
- `paper_results/` contains only aggregate statistics: model metrics summarized over folds, calibration and cohort summaries, and per-procedure or per-code averages, each computed over at least tens of episodes. The case-study figures show model risk trajectories for two episodes with event times relative to surgery only; they contain no identifiers or calendar dates.
- `data/` contains only public vocabulary mappings and the code lists that define the model inputs: CCS category names, SNOMED CT procedure names, RxNorm ingredient names and OMOP laboratory concepts.
- Database credentials are read from environment variables and are never stored in the code.

## Citation

If you use this code, please cite the accompanying paper (citation to be added on publication).
