#!/usr/bin/env bash
set -euo pipefail
port=${1:?port required}
host=${2:-0.0.0.0}
[[ "$port" =~ ^[0-9]+$ ]] || exit 64
exec bash "$(dirname "$0")/guard-large-model.sh" "ds4-0731-$port" 104 \
  --ipc host --shm-size 2g \
  -e DS4_SERVER_COALESCE=0 -e DS4_METAL_PREFILL_CHUNK=1024 \
  -v /home/sparky/LLMs/ollama/DeepSeek/0731:/models:ro \
  -v /home/sparky/LLMs/ollama/DeepSeek/0731-kv:/kv \
  local/ds4-0731:76d51ef \
  --model /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --no-spec --cuda --host "$host" --port "$port" \
  --ctx 131072 --tokens 8192 --mem-floor-gb 12 \
  --kv-disk-dir /kv --kv-disk-space-mb 4096
