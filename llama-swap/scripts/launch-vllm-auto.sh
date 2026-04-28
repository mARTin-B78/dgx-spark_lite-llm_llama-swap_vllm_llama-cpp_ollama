#!/bin/bash
# Generic vLLM launcher with adaptive --gpu-memory-utilization.
#
# - Estimates required VRAM = weights (safetensors on disk) + KV cache + overhead.
# - Queries actually-free VRAM via nvidia-smi (run inside vllm image so it sees
#   the same CUDA view; nvidia-smi on host reports unified memory).
# - Picks the smallest utilization that satisfies the estimate, clamped to
#   [GMEM_MIN, GMEM_MAX]. Fails fast if even GMEM_MAX cannot fit.
#
# Required env:
#   MODEL_PATH       absolute path inside container (e.g. /models/vllm/.../Qwen3.6-35B-A3B-FP8)
#   MODEL_HOST_PATH  same path on host (so this script can stat/read config.json)
#   CONTAINER_NAME   docker --name to use
#   IMAGE            docker image (e.g. vllm-node-tf5:latest)
#   PORT, HOST       passed by llama-swap
#   MAX_MODEL_LEN    context length used for KV estimate (default 131072)
#   MAX_NUM_SEQS     batch size used for KV estimate (default 10)
#   KV_DTYPE_BYTES   1 for fp8, 2 for bf16/fp16 (default 1)
#   GMEM_MIN         floor (default 0.55)
#   GMEM_MAX         ceiling (default 0.92)
#   SAFETY_GIB       headroom on top of estimate (default 4)
#
# Remaining args ($@) are passed verbatim after `vllm serve $MODEL_PATH ...`.

set -euo pipefail

: "${MODEL_PATH:?}"
: "${MODEL_HOST_PATH:?}"
: "${CONTAINER_NAME:?}"
: "${IMAGE:?}"
: "${PORT:?}"
: "${HOST:?}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
KV_DTYPE_BYTES="${KV_DTYPE_BYTES:-1}"
GMEM_MIN="${GMEM_MIN:-0.55}"
GMEM_MAX="${GMEM_MAX:-0.92}"
SAFETY_GIB="${SAFETY_GIB:-4}"

# ── Estimate weights size (sum of *.safetensors) ──────────────────────────────
WEIGHTS_BYTES=$(find "$MODEL_HOST_PATH" -maxdepth 1 -name '*.safetensors' \
    -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
WEIGHTS_GIB=$(awk -v b="$WEIGHTS_BYTES" 'BEGIN{printf "%.2f", b/1073741824}')

# ── Estimate KV cache from config.json ────────────────────────────────────────
CFG="$MODEL_HOST_PATH/config.json"
KV_GIB=$(python3 - "$CFG" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$KV_DTYPE_BYTES" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
ctx, batch, kvb = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# Prefer text_config (multimodal models); fall back to top-level.
src = cfg.get("text_config") or cfg
n_layers = src.get("num_hidden_layers") or src.get("n_layer") or 0
n_heads  = src.get("num_attention_heads") or src.get("n_head") or 1
n_kv     = src.get("num_key_value_heads", n_heads)
hidden   = src.get("hidden_size") or src.get("n_embd") or 0
head_dim = src.get("head_dim") or (hidden // n_heads if n_heads else 0)
# 2 = K + V tensors; covers dense path. MoE/Mamba layers are smaller in practice.
kv_bytes = 2 * n_layers * n_kv * head_dim * ctx * batch * kvb
print(f"{kv_bytes/1073741824:.2f}")
PY
)

NEED_GIB=$(awk -v w="$WEIGHTS_GIB" -v k="$KV_GIB" -v s="$SAFETY_GIB" \
    'BEGIN{printf "%.2f", w+k+s}')

# ── Query VRAM via the same image vLLM will run in ────────────────────────────
MEM_LINE=$(docker run --rm --runtime nvidia --gpus all "$IMAGE" \
    sh -c "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | head -1" \
    2>/dev/null || true)
FREE_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1+0}')
TOTAL_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')

if [ "${FREE_MIB:-0}" -lt 1000 ]; then
    echo "[auto-gmem] WARNING: VRAM query failed (got '${MEM_LINE}'), using GMEM_MIN=${GMEM_MIN}"
    GMEM="$GMEM_MIN"
else
    GMEM=$(awk -v need="$NEED_GIB" -v free_mib="$FREE_MIB" -v total_mib="$TOTAL_MIB" \
            -v gmin="$GMEM_MIN" -v gmax="$GMEM_MAX" 'BEGIN{
        free_gib  = free_mib  / 1024;
        total_gib = total_mib / 1024;
        if (need > free_gib) {
            printf "ERR need=%.2f free=%.2f", need, free_gib;
            exit;
        }
        # request just enough — utilization is computed against CUDA total,
        # which on Spark is ~121.69 GiB vs nvidia-smi 128 GiB; close enough
        # for the cap, since vLLM checks free > util*total at startup.
        u = need / total_gib;
        if (u < gmin) u = gmin;
        if (u > gmax) u = gmax;
        # also cap at what is actually free (minus 1 GiB margin)
        u_cap = (free_gib - 1) / total_gib;
        if (u > u_cap) u = u_cap;
        printf "%.2f", u;
    }')
fi

case "$GMEM" in
    ERR*) echo "[auto-gmem] FATAL: insufficient VRAM — $GMEM GiB"; exit 1 ;;
esac

echo "[auto-gmem] weights=${WEIGHTS_GIB}GiB kv=${KV_GIB}GiB safety=${SAFETY_GIB}GiB → need=${NEED_GIB}GiB"
echo "[auto-gmem] free=$((FREE_MIB/1024))GiB total=$((TOTAL_MIB/1024))GiB → gpu_memory_utilization=${GMEM}"

exec docker run --rm --name "$CONTAINER_NAME" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/sparky/LLMs/vllm:/models/vllm \
    "$IMAGE" \
    vllm serve "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --gpu-memory-utilization "$GMEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "$@"
