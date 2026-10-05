#!/usr/bin/env bash
# Usage: train_eval.sh ARCH [epochs] [batch_size]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"
train_eval_timeseries "${1:?ARCH required (transformer|lstm|grud)}" "${2:-10}" "${3:-}"
