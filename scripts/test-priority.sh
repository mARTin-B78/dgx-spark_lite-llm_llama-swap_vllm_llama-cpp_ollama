#!/usr/bin/env bash
# Priority ordering check under per-key max_parallel_requests=1:
#   P4 holds the GPU (long generation), P3 queues, then P0 queues last.
# Expect: P0 finishes before P3 (admission queue orders by key priority).
# Env: P4_KEY, P3_KEY, P0_KEY, MODEL (required); LITELLM_URL (default http://localhost:4000).
# Body priority is NOT set here: LiteLLM/priority_hook.py derives it from the key's metadata.priority.
set -Eeuo pipefail
: "${P4_KEY:?}" "${P3_KEY:?}" "${P0_KEY:?}" "${MODEL:?}"
url="${LITELLM_URL:-http://localhost:4000}/v1/chat/completions"

req() { # label key max_tokens
  local t0=$SECONDS
  curl -s -o /dev/null --max-time 600 -w "$1 status=%{http_code} elapsed=%{time_total}s\n" "$url" \
    -H "Authorization: Bearer $2" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":$3,\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story.\"}]}"
}

req P4-hold "$P4_KEY" 800 & sleep 3
req P3-queued "$P3_KEY" 32 & sleep 1
req P0-last "$P0_KEY" 32 &
wait
