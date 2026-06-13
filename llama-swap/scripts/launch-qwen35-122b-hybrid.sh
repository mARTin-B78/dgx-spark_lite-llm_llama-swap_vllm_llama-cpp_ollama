#!/bin/bash
# Launch Qwen3.5-122B-A10B hybrid INT4+FP8 checkpoint with MTP-2 speculative decoding.
#
# Prerequisites (run ONCE on host to build the hybrid checkpoint):
#   cd /home/sparky/llama-service/spark-vllm-docker/mods/fix-qwen3.5-hybrid-int4fp8/host
#   pip install torch numpy safetensors huggingface_hub
#   python3 build-hybrid-checkpoint.py \
#     --gptq-dir /home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound \
#     --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 \
#     --output /home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8 \
#     --force
#   python3 add-mtp-weights.py \
#     --source /home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound \
#     --target /home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8
#
# Performance: 28.3 tok/s (baseline) → ~38 tok/s (+35%) with hybrid+MTP-2
# For full v2 (INT8 LM Head): add mods/fix-qwen3.5-hybrid-int4fp8 to the recipe
# Usage (llama-swap cmd): /app/scripts/launch-qwen35-122b-hybrid.sh ${PORT} ${HOST}

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
echo "[122B-hybrid auto-gmem] MemAvailable=${AVAIL_GIB} GiB → gpu_memory_utilization=${GMEM}"

# ── Apply mod and launch ──────────────────────────────────────────────────────
MOD_DIR="/home/sparky/llama-service/spark-vllm-docker/mods/fix-qwen3.5-hybrid-int4fp8"
MODEL_PATH="/home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8"
CONTAINER_NAME="vllm-qwen3.5-122b-hybrid-${PORT}"

docker run --rm --name "${CONTAINER_NAME}" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v "${LLM_ROOT_PATH:-/home/sparky/LLMs/vllm}:/models/vllm" \
    -v "${MOD_DIR}:/opt/mod:ro" \
    vllm-node:latest \
    bash -c "
        cd /opt/mod && ./run.sh &&
        exec vllm serve /models/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8 \
            --served-model-name Qwen3.5-122B-A10B-hybrid-int4fp8 \
            --host '${HOST}' --port '${PORT}' \
            --gpu-memory-utilization '${GMEM}' \
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
            --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' \
            --default-chat-template-kwargs '{\"enable_thinking\": true}'
    "
