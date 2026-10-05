#!/usr/bin/env bash
# Usage: train_eval.sh ARCH [epochs] [batch_size] [gpu]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"
train_eval_event_stream "${1:?ARCH required (transformer|lstm|gru)}" "${2:-20}" "${3:-16}" "${4:-0}"
