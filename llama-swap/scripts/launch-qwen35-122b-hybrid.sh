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

# ── Memory query (same as baseline script) ────────────────────────────────────
MEM_LINE=$(docker run --rm --runtime nvidia --gpus all \
    vllm-node:latest \
    sh -c "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheaders,nounits | head -1" \
    2>/dev/null || true)

FREE_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1+0}')
TOTAL_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')

FREE_NUM=${FREE_MIB:-0}
if [ "${FREE_NUM}" -lt 1000 ] 2>/dev/null; then
    echo "[122B-hybrid auto-gmem] WARNING: VRAM query returned ${FREE_MIB:-empty}, falling back to gmem=0.80"
    GMEM="0.80"
else
    GMEM=$(awk -v f="$FREE_MIB" -v t_nv="$TOTAL_MIB" 'BEGIN {
        cuda_t  = 124610;
        safety  = 3072;
        overhead = (t_nv > cuda_t) ? t_nv - cuda_t : 0;
        cuda_free = f - overhead - safety;
        if (cuda_free < 0) cuda_free = 0;
        u = cuda_free / cuda_t;
        if (u > 0.85) u = 0.85;
        if (u < 0.78) u = 0.78;
        printf "%.2f", u;
    }')
fi

echo "[122B-hybrid auto-gmem] nvidia-smi free=${FREE_MIB} MiB / total=${TOTAL_MIB} MiB → gpu_memory_utilization=${GMEM}"

# ── Apply mod and launch ──────────────────────────────────────────────────────
MOD_DIR="/home/sparky/llama-service/spark-vllm-docker/mods/fix-qwen3.5-hybrid-int4fp8"
MODEL_PATH="/home/sparky/LLMs/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8"
CONTAINER_NAME="vllm-qwen3.5-122b-hybrid-${PORT}"

docker run --rm --name "${CONTAINER_NAME}" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v "${LLM_ROOT_PATH:-/home/sparky/LLMs}/vllm:/models/vllm" \
    -v "${MOD_DIR}:/opt/mod:ro" \
    vllm-node:latest \
    bash -c "
        cd /opt/mod && ./run.sh &&
        exec vllm serve /models/vllm/Alibaba/Qwen3.5-122B-A10B-hybrid-int4fp8 \
            --served-model-name Qwen3.5-122B-A10B-int4-AutoRound \
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
