#!/bin/bash
set -euo pipefail

cd /home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PWD/test-results/overnight-logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/overnight_${TIMESTAMP}.log"

MODELS=(
    "Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4"
    "Qwen3.5-122B-A10B-heretic-v2-NVFP4"
    "Qwen3.5-122B-A10B-NVFP4"
)

log() {
    printf '%s\n' "$*" | tee -a "$MASTER_LOG"
}

on_exit() {
    local status=$?
    log "Sequence finished with exit code $status at $(date '+%Y-%m-%d %H:%M:%S')"
}
trap on_exit EXIT

log "Starting ordered benchmark sequence"
log "Master log: $MASTER_LOG"
log "Model order:"
for model in "${MODELS[@]}"; do
    log "  - $model"
done
log ""

# Run the models one at a time so each start/load/benchmark is easy to inspect.
# benchmark-models.sh already writes detailed report/checkpoint files; the outer
# log here preserves the sequence and the model-by-model console output.
for model in "${MODELS[@]}"; do
    safe_name="${model//\//_}"
    model_log="$LOG_DIR/${safe_name}_${TIMESTAMP}.log"

    log "=================================================="
    log "Benchmarking $model"
    log "Model log: $model_log"
    log "=================================================="

    if ./benchmark-models.sh --quality --quality-mode full "$model" 2>&1 | tee -a "$model_log"; then
        log "Completed $model successfully"
    else
        status=$?
        log "FAILED $model with exit code $status"
    fi
    log ""
done
