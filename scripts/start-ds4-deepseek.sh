#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS4_DIR="${DS4_DIR:-${ROOT_DIR}/ds4}"
MODEL_PATH="${MODEL_PATH:-/home/sparky/LLMs/ollama/DeepSeek/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18004}"
CTX="${CTX:-100000}"
TOKENS="${TOKENS:-8192}"
KV_DIR="${KV_DIR:-/tmp/ds4-kv}"
KV_MB="${KV_MB:-8192}"

exec "${DS4_DIR}/ds4-server" \
  --chdir "${DS4_DIR}" \
  --model "${MODEL_PATH}" \
  --cuda \
  --host "${HOST}" \
  --port "${PORT}" \
  --ctx "${CTX}" \
  --tokens "${TOKENS}" \
  --kv-disk-dir "${KV_DIR}" \
  --kv-disk-space-mb "${KV_MB}" \
  --cors
