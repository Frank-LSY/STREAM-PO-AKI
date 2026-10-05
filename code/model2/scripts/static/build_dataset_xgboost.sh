#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

args=(--folds_dir "${DATA_DIR}" --output_dir "${XGB_DATA_DIR}")
if (($# > 0)); then
  args+=("$@")
fi
"${PYTHON}" "${STATIC_DIR}/xgboost/build_xgboost_dataset.py" "${args[@]}"
