#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

usage() {
    cat <<EOF
Usage: $(basename "$0") [IMAGE...]

Build and push Spark GB10 Docker images to the configured container registry.

  IMAGE   One or more image names to build and push. When omitted, all images
          are built and pushed. Multiple names are space-separated.

Available images:
  llama-cpp-spark   llama.cpp inference engine (GGUF models)
  vllm-spark        vLLM inference engine (Safetensors models)
  llama-swap-spark  llama-swap VRAM orchestrator
  ollama-spark      Ollama model manager
  litellm-spark     LiteLLM API gateway

Examples:
  $(basename "$0")                              # build and push all images
  $(basename "$0") litellm-spark                # build and push one image
  $(basename "$0") litellm-spark vllm-spark     # build and push two images

Registry and credentials are read from .env (REGISTRY, IMAGE_NAMESPACE,
REGISTRY_USER, REGISTRY_TOKEN).
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# --- CONFIGURATION ---
# Resolve the repo root from the script's own location so this works no matter
# where the user cloned the repo (do not assume ~/Docker/...).
# Allow REPO_ROOT to be overridden via env if needed.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# 1. Load variables from .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
    echo "📄 Loaded configuration from .env"
else
    echo "⚠️  No .env file found at $REPO_ROOT/.env"
    exit 1
fi

# --- Registry / namespace (override in .env or via env vars) -----------------
# REGISTRY         where to push (e.g. ghcr.io, registry.gitlab.com,
#                  registry.example.com:5000). Default: ghcr.io
# IMAGE_NAMESPACE  path under the registry that prefixes every image name.
#                  For ghcr.io this is your GitHub username/org. For GitLab
#                  it is "<group>/<project>". Defaults to $GH_USER (back-compat).
# REGISTRY_USER /  credentials used for `docker login`. Default to GH_USER /
# REGISTRY_TOKEN   GITHUB_TOKEN so existing ghcr.io setups keep working unchanged.
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

echo "🚀 Starting Spark GB10 Unified Build Process..."
echo "   Registry  : $REGISTRY"
echo "   Namespace : $IMAGE_NAMESPACE"

# Move to project root
cd "$REPO_ROOT"

# 2. Automated Login (No prompt)
echo "🔑 Logging into container registry: $REGISTRY"
echo "$REGISTRY_TOKEN" | docker login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin

# 3. Build and Push Function
build_and_push() {
    local folder=$1
    local dockerfile=$2
    local image_name=$3

    local tag="$REGISTRY/$IMAGE_NAMESPACE/$image_name:latest"

    echo "---------------------------------------------------------"
    echo "🛠️  Building: $tag"
    echo "---------------------------------------------------------"

    docker build -t "$tag" -f "$folder/$dockerfile" "$folder"

    echo "📦 Pushing: $tag"
    docker push "$tag"
}

# --- EXECUTE BUILDS ---

# If image names are passed as arguments, only build those; otherwise build all.
# Usage: run_if_selected <folder> <dockerfile> <image_name> [filter...]
run_if_selected() {
    local folder=$1 dockerfile=$2 image_name=$3
    shift 3  # remaining args are the filter list (may be empty)
    if [ $# -eq 0 ] || printf '%s\n' "$@" | grep -qx "$image_name"; then
        build_and_push "$folder" "$dockerfile" "$image_name"
    else
        echo "⏭️  Skipping: $image_name"
    fi
}

# Capture the filter list (may be empty)
FILTER=("$@")

# 1. LLAMA-CPP (Optimized for Spark GB10)
run_if_selected "./llama-cpp" "llama-cpp.Dockerfile" "llama-cpp-spark" "${FILTER[@]}"

# 2. VLLM
run_if_selected "./vllm" "vllm.Dockerfile" "vllm-spark" "${FILTER[@]}"

# 3. LLAMA-SWAP
run_if_selected "./llama-swap" "llama-swap.Dockerfile" "llama-swap-spark" "${FILTER[@]}"

# 4. OLLAMA
run_if_selected "./ollama" "ollama.Dockerfile" "ollama-spark" "${FILTER[@]}"

# 5. LITELLM
run_if_selected "./LiteLLM" "litellm.Dockerfile" "litellm-spark" "${FILTER[@]}"

if [ ${#FILTER[@]} -eq 0 ]; then
    echo "✅ All Spark-optimized images have been pushed to $REGISTRY!"
else
    echo "✅ Done pushing: ${FILTER[*]} to $REGISTRY!"
fi
