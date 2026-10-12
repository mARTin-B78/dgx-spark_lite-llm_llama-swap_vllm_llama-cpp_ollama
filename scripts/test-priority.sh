#!/usr/bin/env bash
# Keys stay in the environment, never in process arguments. See --help.
set -Eeuo pipefail
exec python3 "$(dirname "$0")/test_priority.py" "$@"
