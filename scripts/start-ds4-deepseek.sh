#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4_DIR="${DS4_DIR:-${ROOT_DIR}/ds4}"
MODEL_PATH="${MODEL_PATH:-/home/sparky/LLMs/ollama/DeepSeek/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
DSPARK_DRAFTER="${DSPARK_DRAFTER:-${DS4_DIR:-${ROOT_DIR}/ds4}/gguf/DeepSeek-V4-Flash-DSpark-support.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18004}"
CTX="${CTX:-100000}"
TOKENS="${TOKENS:-8192}"
KV_DIR="${KV_DIR:-/tmp/ds4-kv}"
KV_MB="${KV_MB:-8192}"
BATCHED_SESSIONS="${BATCHED_SESSIONS:-1}"
# ENABLE_DSPARK=1 to opt back into speculative decoding. Default is OFF: measured
# real peak usage with DSpark is ~104-113 GiB against a ~121.69 GiB unified pool
# shared with the desktop GPU — margin is too thin and reliably triggers real
# "NVRM: Out of memory" driver errors (verified via dmesg), which can destabilize
# the whole graphical session, not just this process. This is a confirmed capacity
# ceiling (see ds4.c: --gpu-vram budget check runs BEFORE DSpark weights are loaded
# and BEFORE the shared prefill workspace is allocated, so the check can't see
# ~10-12 GiB of the real requirement) — not something further flag-tuning fixes.
# Without DSpark this model fits comfortably (~83-90 GiB) with real headroom.
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
GPU_VRAM="${GPU_VRAM:-88}"

DSPARK_ARGS=()
if [ "${ENABLE_DSPARK}" = "1" ]; then
  DSPARK_ARGS=(--mtp "${DSPARK_DRAFTER}" --dspark)
fi

exec "${DS4_DIR}/ds4-server" \
  --chdir "${DS4_DIR}" \
  --model "${MODEL_PATH}" \
  "${DSPARK_ARGS[@]}" \
  --gpu-vram "${GPU_VRAM}" \
  --batched-session "${BATCHED_SESSIONS}" \
  --cuda \
  --host "${HOST}" \
  --port "${PORT}" \
  --ctx "${CTX}" \
  --tokens "${TOKENS}" \
  --kv-disk-dir "${KV_DIR}" \
  --kv-disk-space-mb "${KV_MB}" \
  --cors
