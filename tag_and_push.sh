#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") [IMAGE...]

Tag locally built vllm-node images and push them to the configured container registry.

  IMAGE   One or more image names to tag and push. When omitted, all images
          are tagged and pushed. Multiple names are space-separated.

Available images:
  vllm-node        Standard vLLM engine (most models: Qwen, Nemotron, Mistral)
  vllm-node-tf5    Transformers v5 variant (Mamba/hybrid models)
  vllm-node-mxfp4  CUTLASS MXFP4 variant (GPT-OSS-120B)

Examples:
  $(basename "$0")                           # tag and push all three images
  $(basename "$0") vllm-node                 # tag and push one image
  $(basename "$0") vllm-node vllm-node-tf5   # tag and push two images

The local images must already exist (built by vllm/build/spark-vllm-docker/build-and-copy.sh).
Registry and credentials are read from .env (REGISTRY, IMAGE_NAMESPACE,
REGISTRY_USER, REGISTRY_TOKEN).
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# --- CONFIGURATION ---
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
    echo "📄 Loaded configuration from .env"
else
    echo "⚠️  No .env file found at $REPO_ROOT/.env"
    exit 1
fi

REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-${GH_USER:-}}"
REGISTRY_USER="${REGISTRY_USER:-${GH_USER:-}}"
REGISTRY_TOKEN="${REGISTRY_TOKEN:-${GITHUB_TOKEN:-}}"

if [ -z "$IMAGE_NAMESPACE" ]; then
    echo "❌ Error: IMAGE_NAMESPACE (or GH_USER) not set in .env"
    exit 1
fi
if [ -z "$REGISTRY_USER" ] || [ -z "$REGISTRY_TOKEN" ]; then
    echo "❌ Error: REGISTRY_USER/REGISTRY_TOKEN (or GH_USER/GITHUB_TOKEN) not set in .env"
    exit 1
fi

echo "🚀 Tagging and pushing vllm-node images..."
echo "   Registry  : $REGISTRY"
echo "   Namespace : $IMAGE_NAMESPACE"

echo "🔑 Logging into container registry: $REGISTRY"
echo "$REGISTRY_TOKEN" | docker login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin

# --- TAG AND PUSH FUNCTION ---
tag_and_push() {
    local local_image=$1
    local remote_name=$2

    local remote_tag="$REGISTRY/$IMAGE_NAMESPACE/$remote_name"

    echo "---------------------------------------------------------"
    echo "🏷️  Tagging: $local_image → $remote_tag"
    echo "---------------------------------------------------------"
    docker tag "$local_image" "$remote_tag"

    echo "📦 Pushing: $remote_tag"
    docker push "$remote_tag"
}

# --- SELECTIVE EXECUTION ---
# Usage: run_if_selected <local_image> <remote_name> <filter_key> [filter...]
run_if_selected() {
    local local_image=$1 remote_name=$2 filter_key=$3
    shift 3  # remaining args are the filter list (may be empty)
    if [ $# -eq 0 ] || printf '%s\n' "$@" | grep -qx "$filter_key"; then
        tag_and_push "$local_image" "$remote_name"
    else
        echo "⏭️  Skipping: $filter_key"
    fi
}

FILTER=("$@")

run_if_selected "vllm-node:latest"       "vllm-node:latest"       "vllm-node"       "${FILTER[@]}"
run_if_selected "vllm-node-tf5:latest"   "vllm-node-tf5:latest"   "vllm-node-tf5"   "${FILTER[@]}"
run_if_selected "vllm-node-mxfp4:latest" "vllm-node-mxfp4:latest" "vllm-node-mxfp4" "${FILTER[@]}"

if [ ${#FILTER[@]} -eq 0 ]; then
    echo "✅ All vllm-node images have been pushed to $REGISTRY!"
else
    echo "✅ Done pushing: ${FILTER[*]} to $REGISTRY!"
fi
