#!/usr/bin/env bash
# Run tool-eval-bench --short against every model served by llama-swap.
# For each: pre-warm with a curl chat (forces model load + JIT compile so
# the bench's own warmup never times out), then run the bench, capture
# the score and the markdown report path. Writes a SUMMARY.txt at the end.

set -uo pipefail

BASE_URL="${BASE_URL:-http://0.0.0.0:28080}"
SESSION_DIR="/home/sparky/.local/share/uv/tools/tool-eval-bench/sessions/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$SESSION_DIR"

echo "=== Session: $SESSION_DIR ==="

# Discover live model names from the OpenAI /v1/models endpoint
MODELS=$(curl -fsS "$BASE_URL/v1/models" \
  | python3 -c 'import sys,json; print("\n".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))')

if [ -z "$MODELS" ]; then
  echo "no models discovered; aborting" >&2
  exit 1
fi

echo "Models to test:"
echo "$MODELS" | sed 's/^/  - /'
echo

SUMMARY="$SESSION_DIR/SUMMARY.txt"
: > "$SUMMARY"

while IFS= read -r MODEL; do
  [ -z "$MODEL" ] && continue
  echo
  echo "=== [$MODEL] $(date -Iseconds) ==="

  WARMUP_LOG="$SESSION_DIR/${MODEL}.warmup.json"
  BENCH_LOG="$SESSION_DIR/${MODEL}.bench.log"

  # 1) Pre-warm — triggers llama-swap to load the model and forces JIT compile.
  #    --max-time 1800s tolerates ~30 min loads (L-tier 120B+ models).
  echo "  warmup ..."
  WT=$(date +%s)
  PAYLOAD=$(MODEL="$MODEL" python3 -c 'import json,os; print(json.dumps({"model":os.environ["MODEL"],"messages":[{"role":"user","content":"warmup"}],"max_tokens":4}))')
  if ! curl -fsS --max-time 1800 "$BASE_URL/v1/chat/completions" \
       -H 'Content-Type: application/json' \
       -d "$PAYLOAD" \
       -o "$WARMUP_LOG"; then
    echo "  WARMUP FAILED after $(( $(date +%s) - WT ))s — skipping bench"
    printf '%-65s %s\n' "$MODEL" "WARMUP_FAILED" >> "$SUMMARY"
    continue
  fi
  echo "  warmup ok in $(( $(date +%s) - WT ))s"

  # 2) Run the bench non-interactively
  echo "  bench  ..."
  BT=$(date +%s)
  tool-eval-bench --base-url "$BASE_URL" --model "$MODEL" --short --no-live \
      > "$BENCH_LOG" 2>&1
  BS=$(( $(date +%s) - BT ))

  SCORE=$(grep -oE 'Score:[[:space:]]+[0-9]+ / 100' "$BENCH_LOG" | tail -1)
  REPORT=$(grep -oE '/home/sparky/[^[:space:]]+\.md' "$BENCH_LOG" | tail -1)

  echo "  ${SCORE:-Score: ?}  (bench ${BS}s)  report=$REPORT"
  printf '%-65s %-22s %s\n' "$MODEL" "${SCORE:-?}" "$REPORT" >> "$SUMMARY"

done <<< "$MODELS"

echo
echo "=========================== SUMMARY ==========================="
cat "$SUMMARY"
echo "==============================================================="
echo "Session dir: $SESSION_DIR"
