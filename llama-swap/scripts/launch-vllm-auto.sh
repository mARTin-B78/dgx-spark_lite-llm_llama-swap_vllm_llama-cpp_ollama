#!/bin/bash
# Generic vLLM launcher with adaptive --gpu-memory-utilization.
#
# - Estimates required VRAM = weights (safetensors on disk) + KV cache + overhead.
# - Reads /proc/meminfo for free/total memory (works on Spark GB10 unified
#   memory, where `nvidia-smi --query-gpu=memory.*` returns "Not Supported").
# - Picks the smallest utilization that satisfies the estimate, clamped to
#   [GMEM_MIN, GMEM_MAX]. Fails fast if even GMEM_MAX cannot fit.
#
# llama-swap container is minimal — uses only awk/sed/grep/find. No python.
#
# Required env:
#   MODEL_PATH       absolute path the spawned vLLM container will use
#                    (e.g. /models/vllm/Alibaba/Qwen3.6-35B-A3B-FP8)
#   MODEL_HOST_PATH  absolute path the launcher script can read
#                    (typically same as MODEL_PATH if /home/sparky/LLMs
#                    is mounted at /models in the llama-swap container)
#   CONTAINER_NAME   docker --name to use
#   IMAGE            docker image (e.g. vllm-node-tf5:latest)
#   PORT, HOST       passed by llama-swap
#   MAX_MODEL_LEN    context length used for KV estimate (default 131072)
#   MAX_NUM_SEQS     batch size used for KV estimate (default 10)
#   KV_DTYPE_BYTES   1 for fp8, 2 for bf16/fp16 (default 1)
#   GMEM_MIN         floor (default 0.55)
#   GMEM_MAX         ceiling (default 0.92)
#   SAFETY_GIB       headroom on top of estimate (default 4)
#   CUDA_OVERHEAD_GIB  GiB of MemTotal not visible to CUDA (default 6.3 for GB10)
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
CUDA_OVERHEAD_GIB="${CUDA_OVERHEAD_GIB:-6.3}"

# ── Estimate weights size (sum of *.safetensors) ──────────────────────────────
WEIGHTS_BYTES=$(find "$MODEL_HOST_PATH" -maxdepth 1 -name '*.safetensors' \
    -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
WEIGHTS_GIB=$(awk -v b="$WEIGHTS_BYTES" 'BEGIN{printf "%.2f", b/1073741824}')

# ── Parse config.json for KV cache parameters (pure shell) ────────────────────
CFG="$MODEL_HOST_PATH/config.json"
if [ ! -f "$CFG" ]; then
    echo "[auto-gmem] FATAL: config.json not found at $CFG"
    exit 1
fi

# Prefer values from "text_config" block (multimodal models nest LLM cfg there);
# fall back to top-level when no text_config exists.
extract() {
    # $1 = key name (e.g. num_hidden_layers)
    local key="$1" val=""
    # Try text_config block first: lines from "text_config": { ... matching close
    val=$(sed -n '/"text_config"[[:space:]]*:[[:space:]]*{/,/^[[:space:]]*}[[:space:]]*,\?[[:space:]]*$/p' "$CFG" \
          | grep -m1 "\"$key\"[[:space:]]*:" \
          | sed -E "s/.*\"$key\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/")
    if [ -z "$val" ]; then
        val=$(grep -m1 "\"$key\"[[:space:]]*:" "$CFG" \
              | sed -E "s/.*\"$key\"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/")
    fi
    echo "${val:-0}"
}

N_LAYERS=$(extract num_hidden_layers)
N_HEADS=$(extract num_attention_heads)
N_KV=$(extract num_key_value_heads)
HIDDEN=$(extract hidden_size)
HEAD_DIM=$(extract head_dim)

# Derive head_dim if missing; default n_kv to n_heads if absent (MHA).
[ "$N_KV" = "0" ] && N_KV="$N_HEADS"
if [ "$HEAD_DIM" = "0" ] && [ "$N_HEADS" != "0" ]; then
    HEAD_DIM=$(awk -v h="$HIDDEN" -v n="$N_HEADS" 'BEGIN{print int(h/n)}')
fi

if [ "$N_LAYERS" = "0" ] || [ "$HEAD_DIM" = "0" ] || [ "$N_KV" = "0" ]; then
    echo "[auto-gmem] FATAL: could not parse layers/heads/head_dim from $CFG"
    exit 1
fi

KV_GIB=$(awk -v L="$N_LAYERS" -v K="$N_KV" -v D="$HEAD_DIM" \
             -v C="$MAX_MODEL_LEN" -v B="$MAX_NUM_SEQS" -v BY="$KV_DTYPE_BYTES" \
    'BEGIN{ printf "%.2f", (2 * L * K * D * C * B * BY) / 1073741824 }')

NEED_GIB=$(awk -v w="$WEIGHTS_GIB" -v k="$KV_GIB" -v s="$SAFETY_GIB" \
    'BEGIN{printf "%.2f", w+k+s}')

# ── Read free/total memory from /proc/meminfo ─────────────────────────────────
# /proc/meminfo gives kB. CUDA sees ~121.69 GiB on the GB10's 124.6 GiB
# unified pool (~6.3 GiB difference for OS/driver), so subtract that overhead
# from total to get CUDA's view.
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
TOTAL_GIB=$(awk -v t="$MEM_TOTAL_KB" -v o="$CUDA_OVERHEAD_GIB" \
    'BEGIN{printf "%.2f", t/1048576 - o}')
FREE_GIB=$(awk -v a="$MEM_AVAIL_KB" -v o="$CUDA_OVERHEAD_GIB" \
    'BEGIN{f=a/1048576 - o; if(f<0)f=0; printf "%.2f", f}')

# ── Compute utilization ───────────────────────────────────────────────────────
GMEM=$(awk -v need="$NEED_GIB" -v free="$FREE_GIB" -v total="$TOTAL_GIB" \
        -v gmin="$GMEM_MIN" -v gmax="$GMEM_MAX" 'BEGIN{
    if (need > free) {
        printf "ERR need=%.2f free=%.2f", need, free;
        exit;
    }
    u = need / total;
    if (u < gmin) u = gmin;
    if (u > gmax) u = gmax;
    u_cap = (free - 1) / total;
    if (u > u_cap) u = u_cap;
    printf "%.2f", u;
}')

case "$GMEM" in
    ERR*) echo "[auto-gmem] FATAL: insufficient VRAM — $GMEM GiB"; exit 1 ;;
esac

echo "[auto-gmem] cfg: layers=$N_LAYERS kv_heads=$N_KV head_dim=$HEAD_DIM ctx=$MAX_MODEL_LEN batch=$MAX_NUM_SEQS kvb=$KV_DTYPE_BYTES"
echo "[auto-gmem] weights=${WEIGHTS_GIB}GiB kv=${KV_GIB}GiB safety=${SAFETY_GIB}GiB → need=${NEED_GIB}GiB"
echo "[auto-gmem] free=${FREE_GIB}GiB total=${TOTAL_GIB}GiB (CUDA view) → gpu_memory_utilization=${GMEM}"

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
