#!/usr/bin/env bash
# Postoperative 6h bucket eval for timeseries models. Usage: eval_postop6h.sh ARCH
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ARCH="${1:?Usage: eval_postop6h.sh ARCH (transformer|lstm|grud)}"
require_ts6h_data
for fold in {1..5}; do
  run_timeseries_postop6h_eval "${ARCH}" "${fold}"
done
aggregate_timeseries_postop6h_summary "${ARCH}"
