#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

for fold in {1..5}; do
  "${PYTHON}" "${EVENT_STREAM_DIR}/check_value.py" \
    --checkpoint "$(event_stream_best_ckpt transformer "${fold}")" \
    --episode_path "${TEST_JSONL}" \
    --output_dir "${RESULT_DIR}/time_mlp_geometry_nocross_fold${fold}" \
    --grid_postop_hour_max 720 \
    --grid_step_hours 6 \
    --pair_sample_size 50000 \
    --delta_hours_list 6,12,24,48,72
done
