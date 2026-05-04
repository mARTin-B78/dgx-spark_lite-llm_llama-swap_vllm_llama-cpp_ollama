#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- CONFIGURATION ---
# Resolve the repo root from the script's own location so this works no matter
# where the user cloned the repo (do not assume ~/Docker/...).
# Allow REPO_ROOT to be overridden via env if needed.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REGISTRY="ghcr.io"

# 1. Load variables from .env if it exists
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
    echo "📄 Loaded configuration from .env"
else
    echo "⚠️  No .env file found at $REPO_ROOT/.env"
    exit 1
fi

# Ensure mandatory variables are present
if [ -z "$GH_USER" ] || [ -z "$GH_PAT" ]; then
    echo "❌ Error: GH_USER or GH_PAT not found in .env"
    exit 1
fi

echo "🚀 Starting Spark GB10 Unified Build Process..."

# Move to project root
cd "$REPO_ROOT"

# 2. Automated Login (No prompt)
echo "🔑 Logging into GitHub Container Registry..."
echo "$GH_PAT" | docker login $REGISTRY -u "$GH_USER" --password-stdin

# 3. Build and Push Function
build_and_push() {
    local folder=$1
    local dockerfile=$2
    local image_name=$3
    
    echo "---------------------------------------------------------"
    echo "🛠️  Building: $image_name"
    echo "---------------------------------------------------------"
    
    docker build -t "$REGISTRY/$GH_USER/$image_name:latest" -f "$folder/$dockerfile" "$folder"
    
    echo "📦 Pushing: $image_name"
    docker push "$REGISTRY/$GH_USER/$image_name:latest"
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

echo "✅ All Spark-optimized images have been pushed to GHCR!"
