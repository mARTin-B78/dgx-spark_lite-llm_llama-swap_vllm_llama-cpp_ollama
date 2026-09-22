#!/usr/bin/env bash
# 64K single-request profile for coexistence with OCR/TTS services on GB10.
set -euo pipefail
port=${1:?port required}
host=${2:-0.0.0.0}
exec bash "$(dirname "$0")/guard-large-model.sh" "qwen38-uncensored-$port" 44 \
  --ipc host --shm-size 2g --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  -e MAX_JOBS=4 -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor -e TRITON_CACHE_DIR=/root/.cache/triton \
  -v /home/sparky/LLMs/qwen38/uncensored-runtime-cache:/root/.cache \
  -v /home/sparky/LLMs/qwen38/Vtuber-plan/Qwen3.8-27B-Uncensored-NVFP4:/models/qwen38:ro \
  -v /home/sparky/LLMs/qwen38/z-lab/Qwen3.8-27B-DFlash2:/models/dflash2:ro \
  sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3 \
  python3 -m sglang.launch_server \
  --model-path /models/qwen38 --served-model-name Qwen3.8-27B-Uncensored-NVFP4-DFlash2 \
  --host "$host" --port "$port" --language-model-only \
  --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.70 --chunked-prefill-size 2048 \
  --context-length 65536 --max-total-tokens 65536 \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 1 --max-mamba-cache-size 8 --cuda-graph-max-bs-decode 1 \
  --speculative-algorithm DFLASH --speculative-draft-model-path /models/dflash2 \
  --speculative-num-draft-tokens 8 --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --tool-call-parser qwen3_coder --stream-interval 1
