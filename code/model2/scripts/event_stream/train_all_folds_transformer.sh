#!/usr/bin/env bash
# Train event-stream Transformer (production / full model) on folds 1–5.
# Latest embedding + loss defaults (see comment block in run_fold).
# 1 GPU: sequential; multiple GPUs: parallel (one job per fold).
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

run_fold() {
  local fold="$1" device="$2" log="$3"
  local save
  save="$(event_stream_ckpt_dir transformer "${fold}")"
  # Production config (2026-09 redesign):
  #   embed: E_code + E_mod + E_etype(med start/end) + gated value_mlp + age_mlp
  #          + time_mlp(hour/24); num_med_exposure=0; dedicated [BASE]
  #   attn:  block (preop bidirectional + postop causal)
  #   loss:  λ_online=1.0, λ_base=0.3, λ_MEM=0.01; sel=0.9·macro+0.1·base
  "${PYTHON}" "${EVENT_STREAM_DIR}/train_online.py" \
    --save_dir "${save}" \
    --folds_dir "${DATA_DIR}/" \
    --val_fold "${fold}" \
    --test_jsonl "${TEST_JSONL}" \
    --wandb_project model2_bert \
    --wandb_run_name "full_v2_fold${fold}" \
    --epochs 100 --lr 1e-4 --lr_step_size 5 --lr_gamma 0.7 \
    --early_stop_patience 10 --early_stop_min_delta 0.00001 \
    --dropout 0.2 \
    --device "${device}" --batch_size 16 \
    --cls_pos_weight 1 \
    --lambda_online 1.0 --lambda_base 0.3 --lambda_mem 0.01 \
    --mem_mask_ratio 0.15 \
    --best_selection_lambda 0.9 \
    --num_med_exposure 0 \
    --use_time_encoding 1 \
    --attention_mode block \
    --baseline_token dedicated \
    > "${log}" 2>&1
}

run_all_folds_with_gpu_policy "train_event_stream_transformer" run_fold
