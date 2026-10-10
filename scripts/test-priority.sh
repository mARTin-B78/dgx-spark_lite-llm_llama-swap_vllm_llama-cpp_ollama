#!/usr/bin/env bash
# 3 P4 requests, then 1 P0 request; prints HTTP status + elapsed per request.
# Env: P4_KEY, P0_KEY (required), MODEL (required), LITELLM_URL (default http://localhost:4000).
# Body priority is NOT set here: LiteLLM/priority_hook.py derives it from the key's metadata.priority.
set -Eeuo pipefail
: "${P4_KEY:?}" "${P0_KEY:?}" "${MODEL:?}"
url="${LITELLM_URL:-http://localhost:4000}/v1/chat/completions"

req() { # label key
  curl -s -o /dev/null --max-time 300 -w "$1 status=%{http_code} elapsed=%{time_total}s\n" "$url" \
    -H "Authorization: Bearer $2" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"Write a short story.\"}]}"
}

req P4-1 "$P4_KEY" & req P4-2 "$P4_KEY" & req P4-3 "$P4_KEY" &
sleep 0.2
req P0 "$P0_KEY"
wait
