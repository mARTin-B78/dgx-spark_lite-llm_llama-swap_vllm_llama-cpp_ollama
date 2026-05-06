#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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
# REGISTRY_TOKEN   GH_PAT so existing ghcr.io setups keep working unchanged.
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-${GH_USER:-}}"
REGISTRY_USER="${REGISTRY_USER:-${GH_USER:-}}"
REGISTRY_TOKEN="${REGISTRY_TOKEN:-${GH_PAT:-}}"

if [ -z "$IMAGE_NAMESPACE" ]; then
    echo "❌ Error: IMAGE_NAMESPACE (or GH_USER) not set in .env"
    exit 1
fi
if [ -z "$REGISTRY_USER" ] || [ -z "$REGISTRY_TOKEN" ]; then
    echo "❌ Error: REGISTRY_USER/REGISTRY_TOKEN (or GH_USER/GH_PAT) not set in .env"
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

# 1. LLAMA-CPP (Optimized for Spark GB10)
build_and_push "./llama-cpp" "llama-cpp.Dockerfile" "llama-cpp-spark"

# 2. VLLM
build_and_push "./vllm" "vllm.Dockerfile" "vllm-spark"

# 3. LLAMA-SWAP
build_and_push "./llama-swap" "llama-swap.Dockerfile" "llama-swap-spark"

# 4. OLLAMA
build_and_push "./ollama" "ollama.Dockerfile" "ollama-spark"

# 5. LITELLM
build_and_push "./LiteLLM" "litellm.Dockerfile" "litellm-spark"

echo "✅ All Spark-optimized images have been pushed to $REGISTRY!"
