#!/usr/bin/env bash
# Usage: evaluate_event_attribution.sh [fold] [max_ep] [gpu]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

FOLD="${1:-1}"
MAX_EP="${2:-30}"
GPU="${3:-$((FOLD - 1))}"
OUT="${RESULT_DIR}/event_stream/transformer/eval_attribution_fold${FOLD}_test"

"${PYTHON}" "${EVENT_STREAM_DIR}/evaluate_event_attribution.py" \
  --checkpoint "$(event_stream_best_ckpt transformer "${FOLD}")" \
  --episode_path "${TEST_JSONL}" \
  --output_dir "${OUT}" \
  --device "$(device_cuda "${GPU}")" \
  --max_episodes "${MAX_EP}" \
  --max_preop_occlusions 60 \
  --max_postop_query_steps 30 \
  --lookback_hours 72 \
  --plot_case_studies

echo "Done: ${OUT}"
