#!/bin/bash
# Dynamically sets --gpu-memory-utilization based on actually-free VRAM at launch time,
# preventing vLLM's startup sanity check from failing when residual CUDA memory
# from previous model containers hasn't been released yet.
#
# Usage (llama-swap cmd): /app/scripts/launch-qwen35-122b.sh ${PORT} ${host}

set -euo pipefail

PORT="${1}"
HOST="${2}"

# ── Memory query ──────────────────────────────────────────────────────────────
# Run nvidia-smi inside the (cached) vllm-node-tf5 image.
# Adds ~3s overhead — negligible vs. the several minutes this model takes to load.
MEM_LINE=$(docker run --rm --runtime nvidia --gpus all \
    vllm-node-tf5:latest \
    sh -c "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheaders,nounits | head -1" \
    2>/dev/null || true)

FREE_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1+0}')
TOTAL_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')

# ── Compute utilization ───────────────────────────────────────────────────────
# DGX Spark GB10: nvidia-smi reports 128 GiB (131072 MiB) unified memory;
# CUDA only sees 121.69 GiB (124610 MiB) — the difference is OS/driver overhead.
# We compute against CUDA's view so the result matches vLLM's startup check.
#   gmem = (free_nv - nvcuda_overhead - safety) / cuda_total
#   clamped to [0.60, 0.85]
GMEM=$(awk -v f="$FREE_MIB" -v t_nv="$TOTAL_MIB" 'BEGIN {
    cuda_t  = 124610;
    safety  = 3072;
    overhead = (t_nv > cuda_t) ? t_nv - cuda_t : 0;
    cuda_free = f - overhead - safety;
    if (cuda_free < 0) cuda_free = 0;
    u = cuda_free / cuda_t;
    if (u > 0.85) u = 0.85;
    if (u < 0.60) u = 0.60;
    printf "%.2f", u;
}')

# Fallback if the query produced garbage
if [ -z "$GMEM" ] || [ "$FREE_MIB" = "0" ]; then
    echo "[122B auto-gmem] WARNING: VRAM query failed, falling back to gmem=0.75"
    GMEM="0.75"
fi

echo "[122B auto-gmem] nvidia-smi free=${FREE_MIB} MiB / total=${TOTAL_MIB} MiB → gpu_memory_utilization=${GMEM}"

# ── Launch ────────────────────────────────────────────────────────────────────
exec docker run --rm --name "vllm-qwen3.5-122b-${PORT}" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/sparky/LLMs/vllm:/models/vllm \
    vllm-node-tf5:latest \
    vllm serve /models/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound \
    --served-model-name Qwen3.5-122B-A10B-int4-AutoRound \
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
    --mamba-ssm-cache-dtype float16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
