# Paper tables

All performance values are percentages on the held-out temporal test set (n = 23,538 surgical episodes; 2,371 with coded PO-AKI), summarized across the five fold-specific models. Dynamic (postoperative) metrics are **macro-averaged**: each metric is computed separately in each of the 120 six-hour windows from 0 to 720 h after surgery and averaged with equal weight per window. Threshold-based metrics use a probability cutoff of 0.5.

Machine-readable versions of every table are in [`../source_data/`](../source_data/).

---

## Table 1. Study cohort characteristics

| Characteristic | Development: no coded PO-AKI (135,393) | Development: coded PO-AKI (8,147) | Test: no coded PO-AKI (21,167) | Test: coded PO-AKI (2,371) |
|---|---|---|---|---|
| **Sex, n (%)** | | | | |
| Female | 95,478 (70.5%) | 3,371 (41.4%) | 14,736 (69.6%) | 984 (41.5%) |
| Male | 39,915 (29.5%) | 4,776 (58.6%) | 6,431 (30.4%) | 1,387 (58.5%) |
| **Age at surgery, median (IQR), years** | 47 (32, 65) | 65 (52, 75) | 45 (32, 65) | 68 (54, 78) |
| **Race, n (%)** | | | | |
| White | 67,434 (49.8%) | 3,765 (46.2%) | 9,211 (43.5%) | 1,018 (42.9%) |
| African American | 25,252 (18.7%) | 1,989 (24.4%) | 4,113 (19.4%) | 512 (21.6%) |
| Asian ᵃ | 5,177 (3.8%) | 243 (3.0%) | 1,112 (5.3%) | 149 (6.3%) |
| Other ᵇ | 436 (0.3%) | 11 (0.1%) | 79 (0.4%) | 10 (0.4%) |
| Unknown | 37,094 (27.4%) | 2,139 (26.3%) | 6,652 (31.4%) | 682 (28.8%) |
| **Surgery type, n (%)** | | | | |
| Abdominal | 89,860 (66.4%) | 5,235 (64.3%) | 15,329 (72.4%) | 1,581 (66.7%) |
| Intracranial | 2,417 (1.8%) | 376 (4.6%) | 497 (2.3%) | 100 (4.2%) |
| Thoracic | 2,790 (2.1%) | 462 (5.7%) | 490 (2.3%) | 144 (6.1%) |
| Joint | 28,361 (20.9%) | 1,122 (13.8%) | 2,895 (13.7%) | 249 (10.5%) |
| Spinal | 9,908 (7.3%) | 599 (7.4%) | 1,686 (8.0%) | 218 (9.2%) |
| Skin | 2,057 (1.5%) | 353 (4.3%) | 270 (1.3%) | 79 (3.3%) |
| **Mean events per episode** | | | | |
| Conditions | 28.28 | 65.42 | 41.19 | 82.82 |
| Medications | 34.65 | 312.18 | 34.04 | 261.37 |
| Laboratory tests | 2.64 | 1.29 | 4.54 | 2.35 |

ᵃ Chinese, Japanese, Korean, Filipino, Vietnamese, Asian Indian. ᵇ American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Other Pacific Islander, Polynesian.

---

## Table 2. Baseline and postoperative dynamic predictive performance

Mean (minimum, maximum) across five fold models.

| Model type | Model | Baseline AUROC | Baseline AUPRC | Dynamic macro-AUROC | Dynamic macro-AUPRC |
|---|---|---|---|---|---|
| Static | Logistic regression | 91.05 (91.02, 91.07) | 61.17 (61.08, 61.26) | — | — |
| | XGBoost | 90.69 (90.57, 90.88) | 60.47 (60.24, 60.68) | — | — |
| Time series | LSTM | 92.42 (92.21, 92.56) | 67.76 (66.61, 68.34) | 93.24 (92.79, 93.59) | 44.49 (41.97, 46.35) |
| | GRU-D | 92.63 (92.51, 92.83) | 69.48 (69.04, 70.28) | 94.23 (93.98, 94.39) | 54.70 (53.37, 55.41) |
| | Time-series Transformer | 93.33 (93.27, 93.41) | 72.88 (72.72, 73.07) | 94.76 (94.59, 94.86) | 52.45 (50.72, 53.85) |
| Event stream | **STREAM** | **94.19** (94.02, 94.32) | **77.14** (76.70, 77.73) | **94.82** (94.72, 94.96) | **58.86** (58.06, 60.23) |

---

## Table 3. Ablation analysis of STREAM

Five-fold means; parentheses give the mean paired within-fold change from the full model in percentage points.

| Category | Ablation | Baseline AUPRC | Dynamic macro-AUPRC |
|---|---|---|---|
| Full model | — | 77.14 | 58.86 |
| Architecture | No time encoding (A1) | 76.37 (−0.77) | 59.27 (+0.41) |
| | Unidirectional attention (A2) | 77.33 (+0.19) | 58.31 (−0.55) |
| | Procedure token as baseline (A3) | 76.91 (−0.24) | 55.95 (−2.91)* |
| Training objective | No baseline loss (A4) | 15.36 (−61.78)*† | 56.39 (−2.47) |
| | No MEM loss (A5) | 76.83 (−0.31) | 58.20 (−0.66) |
| Input modality | No medication events (I1) | 52.49 (−24.65)* | 23.10 (−35.76)* |
| | No laboratory events (I2) | 78.31 (+1.17) | 56.77 (−2.09) |
| | No condition events (I3) | 74.25 (−2.89)* | 46.61 (−12.25)* |
| | Generic surgery token (I4) | 77.29 (+0.15) | 58.20 (−0.66) |
| Interaction | Procedure-as-baseline + generic surgery (A3+I4) | 76.60 (−0.54) | 57.47 (−1.39) |

\* Worse than the full model in all five folds. † The no-baseline-loss model does not train the baseline head; its baseline AUPRC only shows that the head depends on explicit supervision.

---

## Supplementary Table S4. Complete surgery-index (baseline) performance

Mean (minimum, maximum) across five fold models.

| Model type | Model | AUROC | AUPRC | Accuracy | Specificity | Sensitivity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Static | Logistic regression | 91.05 (91.02, 91.07) | 61.17 (61.08, 61.26) | 82.01 (81.72, 82.21) | 81.82 (81.47, 82.08) | 83.71 (83.34, 83.93) | 34.04 (33.66, 34.25) | 48.39 (48.05, 48.60) |
| | XGBoost | 90.69 (90.57, 90.88) | 60.47 (60.24, 60.68) | 85.53 (85.18, 85.81) | 86.38 (85.92, 86.80) | 77.89 (76.97, 78.57) | 39.05 (38.47, 39.50) | 52.02 (51.65, 52.24) |
| Time series | LSTM | 92.42 (92.21, 92.56) | 67.76 (66.61, 68.34) | 93.47 (93.33, 93.62) | 97.94 (97.66, 98.13) | 53.59 (50.44, 55.93) | 74.46 (72.78, 75.86) | 62.29 (60.36, 63.25) |
| | GRU-D | 92.63 (92.51, 92.83) | 69.48 (69.04, 70.28) | 93.58 (93.47, 93.70) | 97.93 (97.81, 98.07) | 54.78 (53.86, 55.50) | 74.78 (73.67, 76.04) | 63.23 (62.72, 63.82) |
| | Time-series Transformer | 93.33 (93.27, 93.41) | 72.88 (72.72, 73.07) | 94.35 (94.32, 94.39) | 98.19 (98.05, 98.28) | 60.08 (59.05, 61.07) | 78.80 (77.81, 79.32) | 68.18 (67.70, 68.43) |
| Event stream | STREAM | 94.19 (94.02, 94.32) | 77.14 (76.70, 77.73) | 94.95 (94.74, 95.14) | 98.36 (97.96, 98.58) | 64.52 (62.67, 66.05) | 81.59 (78.38, 83.42) | 72.03 (71.25, 72.77) |

---

## Supplementary Table S5. Complete postoperative dynamic performance (macro-averaged)

Mean (minimum, maximum) across five fold models. F1 is set to 0 in windows with no true positives at the 0.5 cutoff.

| Model type | Model | AUROC | AUPRC | Accuracy | Specificity | Sensitivity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Time series | LSTM | 93.24 (92.79, 93.59) | 44.49 (41.97, 46.35) | 97.09 (96.90, 97.24) | 98.12 (97.90, 98.32) | 47.76 (45.58, 49.58) | 47.44 (43.22, 51.88) | 45.60 (44.35, 46.96) |
| | GRU-D | 94.23 (93.98, 94.39) | 54.70 (53.37, 55.41) | 97.57 (97.41, 97.62) | 98.52 (98.31, 98.59) | 56.72 (53.87, 59.56) | 54.35 (50.77, 55.76) | 54.25 (53.08, 55.61) |
| | Time-series Transformer | 94.76 (94.59, 94.86) | 52.45 (50.72, 53.85) | 97.70 (97.66, 97.73) | 98.62 (98.58, 98.68) | 54.97 (54.48, 55.31) | 54.33 (53.10, 55.48) | 53.24 (52.69, 54.09) |
| Event stream | STREAM | 94.82 (94.72, 94.96) | 58.86 (58.06, 60.23) | 98.30 (98.12, 98.44) | 99.17 (99.03, 99.34) | 63.86 (60.28, 66.16) | 61.64 (59.06, 65.26) | 61.72 (59.39, 63.12) |

---

## Supplementary Table S7. Complete ablation performance at the surgery index

Mean ± SD across five folds.

| Category | Ablation | AUROC | AUPRC | Accuracy | Specificity | Sensitivity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Full model | STREAM | 94.19 ± 0.13 | 77.14 ± 0.49 | 94.95 ± 0.15 | 98.36 ± 0.28 | 64.52 ± 1.51 | 81.59 ± 2.21 | 72.03 ± 0.59 |
| Architecture | No time encoding (A1) | 94.06 ± 0.15 | 76.37 ± 0.70 | 94.73 ± 0.25 | 98.39 ± 0.50 | 62.11 ± 2.28 | 81.47 ± 4.11 | 70.38 ± 0.67 |
| | Unidirectional attention (A2) | 94.06 ± 0.32 | 77.33 ± 0.89 | 94.93 ± 0.21 | 98.42 ± 0.37 | 63.77 ± 1.36 | 82.06 ± 3.25 | 71.72 ± 0.48 |
| | Procedure token as baseline (A3) | 93.99 ± 0.14 | 76.91 ± 0.43 | 95.04 ± 0.08 | 98.55 ± 0.16 | 63.74 ± 1.03 | 83.13 ± 1.41 | 72.14 ± 0.41 |
| Training objective | No baseline loss (A4) | 47.84 ± 28.20 | 15.36 ± 12.15 | 48.82 ± 31.35 | 47.40 ± 38.92 | 61.50 ± 44.35 | 9.78 ± 7.49 | 16.58 ± 12.23 |
| | No MEM loss (A5) | 94.08 ± 0.20 | 76.83 ± 1.13 | 94.89 ± 0.37 | 98.31 ± 0.56 | 64.38 ± 1.84 | 81.30 ± 4.40 | 71.77 ± 1.17 |
| Input modality | No medication events (I1) | 89.28 ± 0.17 | 52.49 ± 0.54 | 91.28 ± 0.19 | 98.20 ± 0.52 | 29.46 ± 6.28 | 65.22 ± 2.87 | 40.12 ± 6.23 |
| | No laboratory events (I2) | 94.34 ± 0.07 | 78.31 ± 0.28 | 95.17 ± 0.10 | 98.86 ± 0.28 | 62.15 ± 2.85 | 86.10 ± 2.44 | 72.12 ± 1.22 |
| | No condition events (I3) | 92.39 ± 0.15 | 74.25 ± 0.96 | 94.84 ± 0.27 | 98.25 ± 0.32 | 64.35 ± 0.58 | 80.56 ± 2.77 | 71.53 ± 1.03 |
| | Generic surgery token (I4) | 94.20 ± 0.10 | 77.29 ± 0.46 | 94.99 ± 0.18 | 98.43 ± 0.24 | 64.30 ± 0.66 | 82.13 ± 2.13 | 72.11 ± 0.65 |
| Interaction | Procedure-as-baseline + generic surgery (A3+I4) | 94.19 ± 0.16 | 76.60 ± 1.48 | 94.80 ± 0.24 | 98.24 ± 0.29 | 64.12 ± 1.22 | 80.35 ± 2.48 | 71.30 ± 1.05 |

---

## Supplementary Table S8. Complete ablation performance for postoperative dynamic prediction (macro-averaged)

Mean ± SD across five folds. F1 is set to 0 in windows with no true positives at the 0.5 cutoff.

| Category | Ablation | AUROC | AUPRC | Accuracy | Specificity | Sensitivity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Full model | STREAM | 94.82 ± 0.09 | 58.86 ± 0.87 | 98.30 ± 0.12 | 99.17 ± 0.12 | 63.86 ± 2.61 | 61.64 ± 2.29 | 61.72 ± 1.39 |
| Architecture | No time encoding (A1) | 95.32 ± 0.45 | 59.27 ± 0.85 | 98.27 ± 0.17 | 99.16 ± 0.22 | 64.08 ± 2.33 | 59.66 ± 3.50 | 60.26 ± 1.42 |
| | Unidirectional attention (A2) | 94.94 ± 0.29 | 58.31 ± 0.71 | 98.27 ± 0.06 | 99.18 ± 0.07 | 59.63 ± 2.45 | 60.66 ± 1.46 | 59.50 ± 1.20 |
| | Procedure token as baseline (A3) | 94.51 ± 0.29 | 55.95 ± 1.28 | 98.16 ± 0.33 | 99.04 ± 0.36 | 60.35 ± 1.78 | 59.62 ± 4.12 | 59.44 ± 2.40 |
| Training objective | No baseline loss (A4) | 94.48 ± 0.49 | 56.39 ± 2.80 | 97.53 ± 1.15 | 98.36 ± 1.26 | 61.06 ± 3.95 | 54.66 ± 10.81 | 56.38 ± 5.91 |
| | No MEM loss (A5) | 94.82 ± 0.29 | 58.20 ± 1.25 | 98.30 ± 0.19 | 99.18 ± 0.25 | 62.82 ± 2.70 | 62.04 ± 2.92 | 61.64 ± 0.81 |
| Input modality | No medication events (I1) | 88.83 ± 0.34 | 23.10 ± 0.77 | 96.28 ± 0.40 | 98.19 ± 0.57 | 24.41 ± 4.77 | 29.41 ± 4.73 | 25.14 ± 1.75 |
| | No laboratory events (I2) | 94.64 ± 0.37 | 56.77 ± 2.05 | 97.39 ± 0.67 | 98.16 ± 0.78 | 62.46 ± 4.26 | 52.49 ± 7.75 | 56.02 ± 2.81 |
| | No condition events (I3) | 89.58 ± 0.85 | 46.61 ± 0.66 | 95.95 ± 2.13 | 96.60 ± 2.29 | 69.18 ± 3.09 | 36.53 ± 11.27 | 44.41 ± 9.10 |
| | Generic surgery token (I4) | 94.41 ± 0.30 | 58.20 ± 0.95 | 98.35 ± 0.11 | 99.29 ± 0.18 | 60.41 ± 5.39 | 63.24 ± 3.50 | 60.79 ± 1.63 |
| Interaction | Procedure-as-baseline + generic surgery (A3+I4) | 94.72 ± 0.55 | 57.47 ± 1.47 | 98.20 ± 0.45 | 99.09 ± 0.52 | 61.72 ± 3.06 | 61.18 ± 5.20 | 60.58 ± 2.34 |

---

## Supplementary table. Calibration at postoperative landmarks

Mean (minimum to maximum) across five fold models. At each landmark, patients still at risk are scored with the latest available prediction, and the outcome is any future coded PO-AKI within 30 days of surgery.

| Landmark | Model | N at risk | Future PO-AKI, n (%) | Brier score | Brier skill score | Calibration intercept | Calibration slope |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (0 h) | STREAM | 23,535 | 2,368 (10.1) | 0.042 (0.041 to 0.044) | 0.532 (0.514 to 0.546) | -0.16 (-0.34 to -0.00) | 0.68 (0.64 to 0.72) |
| | Time-series Transformer | 23,535 | 2,368 (10.1) | 0.046 (0.046 to 0.047) | 0.492 (0.485 to 0.496) | -0.16 (-0.21 to -0.11) | 0.79 (0.77 to 0.80) |
| | LSTM | 23,535 | 2,368 (10.1) | 0.049 (0.047 to 0.050) | 0.463 (0.452 to 0.478) | 0.00 (-0.12 to 0.19) | 0.83 (0.81 to 0.86) |
| | GRU-D | 23,535 | 2,368 (10.1) | 0.050 (0.049 to 0.050) | 0.450 (0.444 to 0.459) | -0.08 (-0.13 to -0.02) | 0.90 (0.88 to 0.91) |
| | XGBoost | 23,535 | 2,368 (10.1) | 0.105 (0.103 to 0.107) | -0.157 (-0.179 to -0.135) | -1.95 (-1.97 to -1.93) | 0.80 (0.78 to 0.82) |
| | Logistic regression | 23,535 | 2,368 (10.1) | 0.134 (0.133 to 0.136) | -0.478 (-0.498 to -0.466) | -2.40 (-2.42 to -2.39) | 0.76 (0.75 to 0.77) |
| 6 h | STREAM | 23,247 | 2,080 (8.9) | 0.036 (0.034 to 0.037) | 0.564 (0.551 to 0.584) | -0.70 (-0.79 to -0.58) | 0.65 (0.64 to 0.66) |
| | Time-series Transformer | 23,247 | 2,080 (8.9) | 0.111 (0.106 to 0.117) | -0.365 (-0.431 to -0.297) | -2.54 (-2.65 to -2.45) | 0.79 (0.75 to 0.83) |
| | LSTM | 23,247 | 2,080 (8.9) | 0.098 (0.086 to 0.106) | -0.208 (-0.304 to -0.050) | -2.18 (-2.35 to -1.94) | 0.88 (0.83 to 0.92) |
| | GRU-D | 23,247 | 2,080 (8.9) | 0.104 (0.099 to 0.109) | -0.276 (-0.343 to -0.214) | -2.32 (-2.39 to -2.25) | 0.89 (0.84 to 0.93) |
| 12 h | STREAM | 23,164 | 1,997 (8.6) | 0.034 (0.033 to 0.035) | 0.567 (0.554 to 0.585) | -0.75 (-0.86 to -0.64) | 0.66 (0.64 to 0.67) |
| | Time-series Transformer | 23,164 | 1,997 (8.6) | 0.067 (0.063 to 0.068) | 0.154 (0.134 to 0.197) | -1.91 (-1.96 to -1.82) | 0.81 (0.76 to 0.87) |
| | LSTM | 23,164 | 1,997 (8.6) | 0.079 (0.073 to 0.085) | -0.005 (-0.078 to 0.071) | -1.94 (-2.06 to -1.80) | 0.85 (0.79 to 0.89) |
| | GRU-D | 23,164 | 1,997 (8.6) | 0.077 (0.074 to 0.083) | 0.019 (-0.055 to 0.061) | -1.92 (-2.05 to -1.83) | 0.92 (0.88 to 0.96) |
| 24 h | STREAM | 22,983 | 1,816 (7.9) | 0.031 (0.030 to 0.031) | 0.578 (0.567 to 0.590) | -0.71 (-0.83 to -0.61) | 0.68 (0.67 to 0.70) |
| | Time-series Transformer | 22,983 | 1,816 (7.9) | 0.051 (0.048 to 0.053) | 0.303 (0.266 to 0.335) | -1.62 (-1.70 to -1.53) | 0.80 (0.75 to 0.84) |
| | LSTM | 22,983 | 1,816 (7.9) | 0.057 (0.050 to 0.059) | 0.223 (0.189 to 0.311) | -1.52 (-1.69 to -1.27) | 0.79 (0.74 to 0.84) |
| | GRU-D | 22,983 | 1,816 (7.9) | 0.059 (0.057 to 0.060) | 0.195 (0.172 to 0.215) | -1.64 (-1.71 to -1.61) | 0.84 (0.81 to 0.88) |
| 48 h | STREAM | 22,781 | 1,614 (7.1) | 0.028 (0.027 to 0.028) | 0.578 (0.568 to 0.585) | -0.66 (-0.78 to -0.58) | 0.70 (0.68 to 0.72) |
| | Time-series Transformer | 22,781 | 1,614 (7.1) | 0.040 (0.039 to 0.041) | 0.394 (0.382 to 0.405) | -1.29 (-1.31 to -1.25) | 0.77 (0.74 to 0.79) |
| | LSTM | 22,781 | 1,614 (7.1) | 0.049 (0.045 to 0.052) | 0.259 (0.213 to 0.310) | -1.33 (-1.52 to -1.19) | 0.74 (0.71 to 0.77) |
| | GRU-D | 22,781 | 1,614 (7.1) | 0.043 (0.042 to 0.045) | 0.340 (0.315 to 0.360) | -1.27 (-1.34 to -1.21) | 0.88 (0.85 to 0.91) |
| 7 d | STREAM | 22,390 | 1,223 (5.5) | 0.022 (0.021 to 0.022) | 0.579 (0.571 to 0.586) | -0.63 (-0.72 to -0.53) | 0.72 (0.70 to 0.75) |
| | Time-series Transformer | 22,390 | 1,223 (5.5) | 0.030 (0.029 to 0.031) | 0.423 (0.406 to 0.438) | -1.06 (-1.11 to -1.01) | 0.77 (0.75 to 0.78) |
| | LSTM | 22,390 | 1,223 (5.5) | 0.042 (0.037 to 0.044) | 0.193 (0.140 to 0.286) | -1.28 (-1.42 to -1.06) | 0.67 (0.63 to 0.73) |
| | GRU-D | 22,390 | 1,223 (5.5) | 0.032 (0.031 to 0.033) | 0.386 (0.351 to 0.400) | -1.07 (-1.19 to -1.00) | 0.84 (0.82 to 0.87) |
