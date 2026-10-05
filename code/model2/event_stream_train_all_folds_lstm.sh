#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/event_stream/train_all_folds_lstm.sh" "$@"
