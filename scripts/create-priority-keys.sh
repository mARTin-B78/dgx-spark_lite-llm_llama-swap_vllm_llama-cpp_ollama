#!/usr/bin/env bash
# Creates five LiteLLM virtual keys; prints each key ONCE to stdout (store in KeePass, never in files).
# metadata.priority: higher = more important. LiteLLM/priority_hook.py maps it to the request-body
# priority (100 - value; lower = served first) and overwrites any client-supplied value.
set -Eeuo pipefail
: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in the environment}"
url="${LITELLM_URL:-http://localhost:4000}"

gen() { # alias priority max_parallel(or null)
  curl -fsS --max-time 15 -X POST "$url/key/generate" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
    -d "{\"key_alias\":\"$1\",\"metadata\":{\"priority\":$2},\"max_parallel_requests\":$3}" |
    python3 -c 'import sys,json;print(json.load(sys.stdin)["key"])'
}

echo "P0-Realtime: $(gen P0-Realtime 100 null)"
echo "P1-Interactive: $(gen P1-Interactive 80 8)"
echo "P2-Normal: $(gen P2-Normal 60 4)"
echo "P3-Automation: $(gen P3-Automation 40 2)"
echo "P4-Background: $(gen P4-Background 20 1)"
