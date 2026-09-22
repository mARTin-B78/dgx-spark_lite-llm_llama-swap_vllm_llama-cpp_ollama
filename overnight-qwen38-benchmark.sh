#!/usr/bin/env bash
# Overnight A/B benchmark for the two Qwen3.8 recipes on one DGX Spark.
#
# Runs serially so the models do not compete for unified memory:
#   1. warm-up/load through llama-swap
#   2. llama-benchy throughput profile via tool-eval-bench
#   3. public tool-eval-bench suite with reasoning enabled
#   4. hard-mode tool-eval-bench suite with reasoning enabled
#   5. optional GSM8K and speculative-decoding live probe
#   6. stop the model before moving to the next one
#
# Start detached for an overnight run:
#   nohup ./overnight-qwen38-benchmark.sh > test-results/qwen38-overnight.nohup.log 2>&1 &
#
# Results are written under test-results/qwen38-overnight/<timestamp>/.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://127.0.0.1:28080}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/test-results/qwen38-overnight}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RESULT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

TOOL_EVAL_BIN="${TOOL_EVAL_BIN:-tool-eval-bench}"
TOOL_EVAL_VERSION="$($TOOL_EVAL_BIN --version 2>&1 || true)"

MODELS=(
  "Qwen3.8-Flash-Next-NVFP4"
  "Qwen3.8-27B-NVFP4-DFlash2"
)

# Forum-style Qwen reasoning sampler. Omitting --no-think keeps reasoning enabled.
REASONING_ARGS=(
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --backend-kwargs '{"reasoning_effort":"medium"}'
  --seed 42
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$RUN_DIR/master.log"
}

snapshot() {
  local name="$1"
  {
    echo "===== $name ====="
    date '+%F %T %Z'
    free -h
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    docker stats --no-stream --format 'container={{.Name}} mem={{.MemUsage}} cpu={{.CPUPerc}}' 2>/dev/null || true
  } > "$RUN_DIR/${CURRENT_MODEL//[^A-Za-z0-9_.-]/_}__${name}.txt"
}

wait_for_model() {
  local model="$1"
  local deadline=$((SECONDS + ${MODEL_READY_TIMEOUT:-1800}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 10 "$LLAMA_SWAP_URL/v1/models" \
      | jq -e --arg model "$model" '.data[]? | select(.id == $model)' >/dev/null 2>&1; then
      # A tiny non-streaming request confirms the selected upstream is usable.
      if curl -fsS --max-time 30 "$LLAMA_SWAP_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -cn --arg model "$model" \
          '{model:$model,messages:[{role:"user",content:"Reply with exactly READY"}],max_tokens:8,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')" \
        | jq -e '.choices[0].message' >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 10
  done
  return 1
}

stop_active_model() {
  # llama-swap starts model containers on demand. Stop only its sglang model
  # containers, leaving llama-swap, LiteLLM, and unrelated services intact.
  mapfile -t containers < <(docker ps --format '{{.Names}}' | grep -E '^sglang-' || true)
  if ((${#containers[@]})); then
    log "Stopping active model container(s): ${containers[*]}"
    docker stop "${containers[@]}" >/dev/null
    sleep 10
  fi
}

run_eval() {
  local label="$1"
  shift
  local outfile="$RUN_DIR/${CURRENT_MODEL//[^A-Za-z0-9_.-]/_}__${label}.json"
  log "Starting $label for $CURRENT_MODEL"
  "$TOOL_EVAL_BIN" \
    --base-url "$LLAMA_SWAP_URL/v1" \
    --model "$CURRENT_MODEL" \
    --timeout "${TOOL_TIMEOUT:-180}" \
    --no-live \
    --json-file "$outfile" \
    --output-dir "$RUN_DIR/reports" \
    --label "qwen38-overnight-$label" \
    "${REASONING_ARGS[@]}" "$@" \
    2>&1 | tee "$RUN_DIR/${CURRENT_MODEL//[^A-Za-z0-9_.-]/_}__${label}.log"
}

CURRENT_MODEL=""
trap 'status=$?; log "Run finished with exit code $status"; exit "$status"' EXIT

command -v curl >/dev/null || { echo 'curl is required' >&2; exit 2; }
command -v jq >/dev/null || { echo 'jq is required' >&2; exit 2; }
command -v docker >/dev/null || { echo 'docker is required' >&2; exit 2; }
command -v "$TOOL_EVAL_BIN" >/dev/null || { echo "$TOOL_EVAL_BIN not found" >&2; exit 2; }

log "Qwen3.8 overnight benchmark started"
log "Tool-eval version: $TOOL_EVAL_VERSION"
log "Endpoint: $LLAMA_SWAP_URL"
log "Results: $RUN_DIR"
log "Models run serially: ${MODELS[*]}"

for CURRENT_MODEL in "${MODELS[@]}"; do
  safe_model="${CURRENT_MODEL//[^A-Za-z0-9_.-]/_}"
  log "============================================================"
  log "Preparing $CURRENT_MODEL"
  snapshot idle-before

  if ! wait_for_model "$CURRENT_MODEL"; then
    log "FAILED to load or reach $CURRENT_MODEL"
    snapshot load-failed
    stop_active_model
    continue
  fi

  snapshot loaded

  # 2048-token prompt, 1024-token generation, common concurrency/depth points.
  # generation latency is preferred so the report separates decode from API time.
  run_eval throughput \
    --perf-only \
    --pp 2048 \
    --tg 1024 \
    --depth 0,4096,8192 \
    --concurrency 1,2,4 \
    --benchy-runs 3 \
    --benchy-latency-mode generation \
    --skip-coherence \
    || log "WARNING: throughput benchmark failed for $CURRENT_MODEL"

  # Default public suite (the current tool-eval release's full suite).
  run_eval tool-suite \
    || log "WARNING: standard tool suite failed for $CURRENT_MODEL"

  # Adds the adversarial Hard Mode scenarios; current releases produce the
  # comparable expanded suite used by the NVIDIA forum reports.
  run_eval hardmode \
    --hardmode \
    || log "WARNING: hard-mode tool suite failed for $CURRENT_MODEL"

  # Common supplementary quality check, kept separate from tool-call scoring.
  run_eval gsm8k \
    --gsm8k-only \
    || log "WARNING: GSM8K benchmark failed for $CURRENT_MODEL"

  snapshot completed
  stop_active_model
done

log "All Qwen3.8 overnight benchmarks completed"
