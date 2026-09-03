#!/usr/bin/env bash
# Prepare Qwen3.8-Flash-Next for one DGX Spark.
# This downloads ~135 GB and builds the required ~12.8 GB HashK PLE artifact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASHNEXT_REPO="${FLASHNEXT_REPO:-/home/sparky/flashnext-one-spark}"
HF_CACHE="${QWEN38_FLASHNEXT_HF_CACHE:-/home/sparky/LLMs/qwen38/flashnext-hf}"
IMAGE="lmsysorg/sglang:qwen38flashnext"

command -v hf >/dev/null || { echo "ERROR: install huggingface_hub first" >&2; exit 1; }
[[ -d "$FLASHNEXT_REPO" ]] || git clone https://github.com/deathbyorderfill/flashnext-one-spark "$FLASHNEXT_REPO"
mkdir -p "$HF_CACHE"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --repo-type model --cache-dir "$HF_CACHE"

docker pull "$IMAGE"
docker run --rm --gpus all \
  -v "$HF_CACHE":/root/.cache/huggingface:ro \
  -v "$FLASHNEXT_REPO":/out \
  --entrypoint python3 "$IMAGE" /out/tools/build_hashk_ple.py

[[ -s "$FLASHNEXT_REPO/ple_hashk_R4.pt" ]] || {
  echo "ERROR: HashK artifact was not created" >&2
  exit 1
}
echo "Flash-Next prepared. Select model: Qwen3.8-Flash-Next-NVFP4"
