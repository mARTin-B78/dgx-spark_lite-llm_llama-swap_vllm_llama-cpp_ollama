#!/usr/bin/env bash
set -euo pipefail
port=${1:?port required}
host=${2:-0.0.0.0}
[[ "$port" =~ ^[0-9]+$ ]] || exit 64
# Existing RadixArk checkpoint; PLE FP8 shards are read from NVMe by B12X.
# No HashK compression, no additional full checkpoint, no MTP draft initially.
exec bash "$(dirname "$0")/guard-large-model.sh" "qwen38-ssd-$port" 104 \
  --ipc host --shm-size 4g --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/sparky/LLMs/qwen38/flashnext-hf:/hf:ro \
  -v /home/sparky/LLMs/qwen38/ssd-runtime-cache:/root/.cache \
  -e VLLM_PLE_TABLE_MEMORY=disk -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CUTE_DSL_ARCH=sm_121a -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e B12X_POLICY_MODE=auto \
  eugr/spark-vllm-b12x@sha256:8e7e062186f841453ef0ec6f713043c5b65447decc3835206685128c18e42262 \
  vllm serve /hf/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --host "$host" --port "$port" --trust-remote-code --language-model-only \
  --dtype bfloat16 --load-format b12x --kv-cache-dtype fp8 \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.74 --kv-cache-memory 4294967296 --enforce-eager \
  --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x \
  --no-enable-flashinfer-autotune --enable-chunked-prefill \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
