#!/bin/bash
# Thin CLI wrapper around recipe_tool.py — run from the repo root.
#   scripts/recipe.sh export <model-name>
#   scripts/recipe.sh import recipes/<model-name>.yaml [--apply]
#   scripts/recipe.sh list
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/recipe_tool.py" "$@"
