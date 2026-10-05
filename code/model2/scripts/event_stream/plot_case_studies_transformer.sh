#!/usr/bin/env bash
# Usage: plot_clinical_case_studies.sh [fold] [attr] [gpu]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

FOLD="${1:-1}"
WITH_ATTR="${2:-}"
GPU="${3:-$((FOLD - 1))}"
OUT="${RESULT_DIR}/event_stream/transformer/case_studies_fold${FOLD}"

extra=()
if [[ "${WITH_ATTR}" == "attr" ]]; then
  extra+=(--with_attribution)
fi

"${PYTHON}" "${EVENT_STREAM_DIR}/plot_clinical_case_studies.py" \
  --checkpoint "$(event_stream_best_ckpt transformer "${FOLD}")" \
  --episode_path "${TEST_JSONL}" \
  --output_dir "${OUT}" \
  --device "$(device_cuda "${GPU}")" \
  --alert_tau 0.5 \
  --dpi 150 \
  "${extra[@]}"

echo "Outputs: ${OUT}/case_study_aki_*.png"
