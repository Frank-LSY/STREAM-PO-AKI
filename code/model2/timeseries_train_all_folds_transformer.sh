#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/timeseries/train_all_folds_transformer.sh" "$@"
