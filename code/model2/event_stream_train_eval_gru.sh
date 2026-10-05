#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/event_stream/train_eval_gru.sh" "$@"
