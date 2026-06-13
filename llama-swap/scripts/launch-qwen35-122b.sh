#!/bin/bash
# Dynamically sets --gpu-memory-utilization based on actually-free VRAM at launch time,
# preventing vLLM's startup sanity check from failing when residual CUDA memory
# from previous model containers hasn't been released yet.
#
# Usage (llama-swap cmd): /app/scripts/launch-qwen35-122b.sh ${PORT} ${host}

set -euo pipefail

PORT="${1}"
HOST="${2}"

# ── Memory query via /proc/meminfo ────────────────────────────────────────────
# nvidia-smi memory.free returns "Not Supported" on GB10 unified memory.
# /proc/meminfo in the llama-swap container reflects the host unified pool.
# CUDA sees ~121.69 GiB of the 128 GiB pool; subtract OS/driver overhead and a
# 5 GiB buffer for the MemAvailable→cudaMemGetInfo race at vLLM startup.
# Floor 0.55 accommodates the always-on 4B service + TTS consuming ~48 GiB.
MEM_AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)

GMEM=$(awk -v a="${MEM_AVAIL_KB:-0}" -v t="${MEM_TOTAL_KB:-134217728}" 'BEGIN {
    cuda_total  = 121.69;
    avail_gib   = a / 1048576;
    mem_total   = t / 1048576;
    overhead    = mem_total - cuda_total;
    if (overhead < 0) overhead = 0;
    free_gib    = avail_gib - overhead - 5;
    if (free_gib < 0) free_gib = 0;
    u = free_gib / cuda_total;
    if (u > 0.85) u = 0.85;
    if (u < 0.55) u = 0.55;
    printf "%.2f", u;
}')

AVAIL_GIB=$(awk -v k="${MEM_AVAIL_KB:-0}" 'BEGIN{printf "%.1f", k/1048576}')
echo "[122B auto-gmem] MemAvailable=${AVAIL_GIB} GiB → gpu_memory_utilization=${GMEM}"

# ── Launch ────────────────────────────────────────────────────────────────────
exec docker run --rm --name "vllm-qwen3.5-122b-${PORT}" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v "${LLM_ROOT_PATH:-/home/sparky/LLMs/vllm}:/models/vllm" \
    vllm-node:latest \
    vllm serve /models/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound \
    --served-model-name Qwen3.5-122B-A10B-int4-AutoRound \
    --chat-template /models/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound/chat_template-tool-strict.jinja \
    --host "${HOST}" --port "${PORT}" \
    --gpu-memory-utilization "${GMEM}" \
    --max-model-len 131072 \
    --max-num-seqs 10 \
    --max-num-batched-tokens 32768 \
    --max-cudagraph-capture-size 10 \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --trust-remote-code \
    --attention-backend FLASHINFER \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}'
