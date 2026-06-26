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
#   SYSTEM_RAM_CEILING_GIB
#                    hard cap on total system RAM the launcher will plan
#                    against. Defaults to 117.81 GiB (= 126.5 GB decimal),
#                    because GB10 unified-memory systems crash near
#                    126.5 GB of used RAM. The launcher leaves
#                    (MemTotal - SYSTEM_RAM_CEILING_GIB) GiB always free.
#                    Set higher only if your system tolerates more pressure.
#
# Optional env:
#   GMEM_OVERRIDE      Either a numeric value (e.g. "0.40", "0.7069") to bypass
#                      the adaptive calculation entirely and pin gpu_memory_utilization
#                      to that value, or "adaptive" / unset / empty to compute it
#                      dynamically based on free VRAM. Use this single knob to
#                      switch a model between static and adaptive without
#                      restructuring the model block.
#   EXTRA_DOCKER_ARGS  extra `docker run` args, space-separated (mounts, envs).
#                      Example: "-v /host/mod:/mod:ro -e VLLM_USE_FLASHINFER_MOE_FP8=1"
#   PRE_LAUNCH_CMD     bash command to execute inside the container before
#                      `vllm serve`. When set, the launcher uses
#                      `--entrypoint /bin/bash -c "PRE_LAUNCH_CMD && exec vllm serve …"`.
#                      Use for image patches, mod scripts, or env setup.
#   VLLM_SERVE_PREFIX  command tokens to invoke vllm-serve. Default "vllm serve".
#                      Set to "" (empty) for images whose ENTRYPOINT already is
#                      `vllm serve` (e.g. vllm/vllm-openai), so the model path
#                      and flags are passed directly as the entrypoint args.
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
SYSTEM_RAM_CEILING_GIB="${SYSTEM_RAM_CEILING_GIB:-117.81}"
# Realistic concurrent-prefill cap for the KV estimate. vLLM scales KV
# dynamically, so estimating against the full MAX_NUM_SEQS overstates the
# need by 5-10× for typical workloads. Default 4 keeps the gmem decision
# in a useful range without blocking the launch.
KV_BATCH_REALISTIC="${KV_BATCH_REALISTIC:-4}"

# ── Static-override fast path ─────────────────────────────────────────────────
# When GMEM_OVERRIDE is a number, skip the entire adaptive calculation and
# pin gpu_memory_utilization to that value. Anything else (unset, empty,
# "adaptive") falls through to the dynamic path below.
GMEM_OVERRIDE="${GMEM_OVERRIDE:-}"
if [ -n "$GMEM_OVERRIDE" ] && [ "$GMEM_OVERRIDE" != "adaptive" ]; then
    if echo "$GMEM_OVERRIDE" | awk '{exit !($0+0 > 0 && $0+0 < 1)}'; then
        GMEM_VAL="$GMEM_OVERRIDE"
        GMEM_MODE="static-override"
        echo "[auto-gmem] GMEM_OVERRIDE=$GMEM_OVERRIDE — bypassing adaptive calculation"
        GMEM="$GMEM_VAL"
        SKIP_ADAPTIVE=1
    else
        echo "[auto-gmem] FATAL: GMEM_OVERRIDE='$GMEM_OVERRIDE' is not a number in (0,1) and not 'adaptive'"
        exit 1
    fi
else
    SKIP_ADAPTIVE=0
fi

if [ "$SKIP_ADAPTIVE" = "0" ]; then

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

KV_BATCH=$(awk -v a="$MAX_NUM_SEQS" -v b="$KV_BATCH_REALISTIC" \
    'BEGIN{ printf "%d", (a < b ? a : b) }')
KV_GIB=$(awk -v L="$N_LAYERS" -v K="$N_KV" -v D="$HEAD_DIM" \
             -v C="$MAX_MODEL_LEN" -v B="$KV_BATCH" -v BY="$KV_DTYPE_BYTES" \
    'BEGIN{ printf "%.2f", (2 * L * K * D * C * B * BY) / 1073741824 }')

NEED_GIB=$(awk -v w="$WEIGHTS_GIB" -v k="$KV_GIB" -v s="$SAFETY_GIB" \
    'BEGIN{printf "%.2f", w+k+s}')

# ── Read free/total memory from /proc/meminfo ─────────────────────────────────
# /proc/meminfo gives kB. CUDA sees ~121.69 GiB on the GB10's 124.6 GiB
# unified pool (~6.3 GiB difference for OS/driver), so subtract that overhead
# from total to get CUDA's view.
#
# Then enforce SYSTEM_RAM_CEILING_GIB: GB10 systems crash near 126.5 GB
# (~117.81 GiB) of used RAM, so we leave (MemTotal - ceiling) GiB always
# free even if /proc/meminfo says more is available. This caps both the
# total budget AND the free-memory window the gmem calculation sees.
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
MEM_FREE_KB=$(awk '/^MemFree:/{print $2}' /proc/meminfo)

MEM_TOTAL_GIB_RAW=$(awk -v t="$MEM_TOTAL_KB" 'BEGIN{printf "%.2f", t/1048576}')
MEM_AVAIL_GIB_RAW=$(awk -v a="$MEM_AVAIL_KB" 'BEGIN{printf "%.2f", a/1048576}')
MEM_FREE_GIB_RAW=$(awk -v f="$MEM_FREE_KB" 'BEGIN{printf "%.2f", f/1048576}')

# Reserved headroom = MemTotal - ceiling (clamped to 0).
HEADROOM_GIB=$(awk -v t="$MEM_TOTAL_GIB_RAW" -v c="$SYSTEM_RAM_CEILING_GIB" \
    'BEGIN{h=t-c; if(h<0)h=0; printf "%.2f", h}')

# Effective MemTotal for the calculation = min(actual MemTotal, ceiling).
MEM_TOTAL_GIB_CAPPED=$(awk -v t="$MEM_TOTAL_GIB_RAW" -v c="$SYSTEM_RAM_CEILING_GIB" \
    'BEGIN{m=(t<c?t:c); printf "%.2f", m}')

TOTAL_GIB=$(awk -v t="$MEM_TOTAL_GIB_CAPPED" -v o="$CUDA_OVERHEAD_GIB" \
    'BEGIN{printf "%.2f", t - o}')

# Use MemAvailable for the free estimate — the kernel reclaims page cache
# as vLLM allocates its CUDA pool, so the "could be freed" figure is what
# the model actually has access to. MemFree alone is too pessimistic after
# a recent model swap (page cache holds the prior weights).
#
# However, vLLM's startup check uses cudaMemGetInfo which sees a snapshot
# CLOSER to MemFree before reclaim. To bridge that race, GMEM_FREE_BUFFER_GIB
# (default 5) is subtracted from u_cap so the launcher never asks for more
# than (MemAvailable - buffer) GiB. Tested fix for the ~1 GiB cudaMemGetInfo
# vs MemAvailable discrepancy that crashed Nemotron-Super-120B.
FREE_GIB=$(awk -v a="$MEM_AVAIL_GIB_RAW" -v h="$HEADROOM_GIB" -v o="$CUDA_OVERHEAD_GIB" \
    'BEGIN{f=a - h - o; if(f<0)f=0; printf "%.2f", f}')

# ── Compute utilization ───────────────────────────────────────────────────────
# Don't fail-fast when need > free: the KV estimate is a worst-case cap, not
# what vLLM actually pre-allocates. vLLM resizes the KV pool dynamically based
# on real concurrent batch and observed free memory. So when our estimate
# exceeds free, fall back to GMEM_MAX (clamped to free) and let vLLM's startup
# check decide. If even GMEM_MIN doesn't fit, that's a true OOM — exit then.
GMEM_FREE_BUFFER_GIB="${GMEM_FREE_BUFFER_GIB:-5}"
GMEM=$(awk -v need="$NEED_GIB" -v free="$FREE_GIB" -v total="$TOTAL_GIB" \
        -v gmin="$GMEM_MIN" -v gmax="$GMEM_MAX" -v buf="$GMEM_FREE_BUFFER_GIB" 'BEGIN{
    # Buffer between launcher cap and CUDA-visible free, since MemAvailable
    # and cudaMemGetInfo can disagree by ~1 GiB right at vLLM startup
    # (kernel reclaims page cache during cudaMalloc, not before the check).
    u_cap = (free - buf) / total;
    if (u_cap < 0) u_cap = 0;
    if (u_cap < gmin) {
        printf "ERR cap=%.2f gmin=%.2f free=%.2f", u_cap, gmin, free;
        exit;
    }
    if (need <= free) {
        u = need / total;
        if (u < gmin) u = gmin;
        if (u > gmax) u = gmax;
        if (u > u_cap) u = u_cap;
        printf "%.2f|sized", u;
    } else {
        u = gmax;
        if (u > u_cap) u = u_cap;
        if (u < gmin) u = gmin;
        printf "%.2f|fallback", u;
    }
}')

case "$GMEM" in
    ERR*)
        echo "[auto-gmem] FATAL: free VRAM below GMEM_MIN floor — $GMEM"
        exit 1 ;;
esac

GMEM_VAL="${GMEM%%|*}"
GMEM_MODE="${GMEM##*|}"

echo "[auto-gmem] cfg: layers=$N_LAYERS kv_heads=$N_KV head_dim=$HEAD_DIM ctx=$MAX_MODEL_LEN batch=$MAX_NUM_SEQS kvb=$KV_DTYPE_BYTES kv_batch_used=$KV_BATCH"
echo "[auto-gmem] weights=${WEIGHTS_GIB}GiB kv=${KV_GIB}GiB safety=${SAFETY_GIB}GiB → need=${NEED_GIB}GiB"
echo "[auto-gmem] system: MemTotal=${MEM_TOTAL_GIB_RAW}GiB MemFree=${MEM_FREE_GIB_RAW}GiB MemAvail=${MEM_AVAIL_GIB_RAW}GiB ceiling=${SYSTEM_RAM_CEILING_GIB}GiB headroom_reserved=${HEADROOM_GIB}GiB"
echo "[auto-gmem] free=${FREE_GIB}GiB total=${TOTAL_GIB}GiB (CUDA view, capped) → gpu_memory_utilization=${GMEM_VAL} [${GMEM_MODE}]"
if [ "$GMEM_MODE" = "fallback" ]; then
    echo "[auto-gmem] NOTE: estimate exceeded free VRAM; using clamped gmax — vLLM will trim KV at startup if needed"
fi
GMEM="$GMEM_VAL"

fi   # end if SKIP_ADAPTIVE == 0

# Common docker args (network, GPU, base mounts/envs).
DOCKER_BASE=(
    docker run --rm --name "$CONTAINER_NAME"
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1
    -v "${LLM_ROOT_PATH:-/home/user/LLMs}/vllm:/models/vllm"
    -v "${LLM_ROOT_PATH:-/home/user/LLMs}/.cache/triton:/root/.triton"
    -v "${LLM_ROOT_PATH:-/home/user/LLMs}/.cache/vllm:/root/.cache/vllm"
)

# Optional: extra mounts / envs (split on whitespace; preserves order).
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
EXTRA_ARR=()
[ -n "$EXTRA_DOCKER_ARGS" ] && read -r -a EXTRA_ARR <<< "$EXTRA_DOCKER_ARGS"

# Common vllm args. VLLM_SERVE_PREFIX defaults to "vllm serve" but can be
# unset (empty) for images whose ENTRYPOINT is already vllm serve.
VLLM_SERVE_PREFIX="${VLLM_SERVE_PREFIX-vllm serve}"
VLLM_PREFIX_ARR=()
[ -n "$VLLM_SERVE_PREFIX" ] && read -r -a VLLM_PREFIX_ARR <<< "$VLLM_SERVE_PREFIX"
VLLM_ARGS=(
    "${VLLM_PREFIX_ARR[@]}"
    "$MODEL_PATH"
    --host "$HOST" --port "$PORT"
    --gpu-memory-utilization "$GMEM"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    "$@"
)

if [ -n "${PRE_LAUNCH_CMD:-}" ]; then
    # Build the full command-line string vllm will receive after PRE_LAUNCH_CMD.
    # Each token is shell-quoted to survive the bash -c evaluation.
    VLLM_QUOTED=""
    for tok in "${VLLM_ARGS[@]}"; do
        VLLM_QUOTED+=" $(printf '%q' "$tok")"
    done
    echo "[auto-gmem] using PRE_LAUNCH_CMD: $PRE_LAUNCH_CMD"
    exec "${DOCKER_BASE[@]}" "${EXTRA_ARR[@]}" \
        --entrypoint /bin/bash \
        "$IMAGE" \
        -c "$PRE_LAUNCH_CMD && exec$VLLM_QUOTED"
else
    exec "${DOCKER_BASE[@]}" "${EXTRA_ARR[@]}" "$IMAGE" "${VLLM_ARGS[@]}"
fi
