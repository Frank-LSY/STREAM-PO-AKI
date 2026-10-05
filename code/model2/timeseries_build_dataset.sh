#!/usr/bin/env bash
exec "$(dirname "$0")/scripts/timeseries/build_dataset.sh" "$@"
