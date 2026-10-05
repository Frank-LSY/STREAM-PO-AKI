#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/timeseries/train_eval_lstm.sh" "$@"
