#!/bin/bash

###############################################################################
# DGX Spark LLM Stack - Model Download Script
# 
# This script automatically downloads models based on your tier selections
# from .env. It uses hf download for efficient, resumable downloads.
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

###############################################################################
# LOAD CONFIGURATION
###############################################################################

# Find the repo root (two levels up from setup/)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [ ! -f "$REPO_ROOT/.env" ]; then
    print_error ".env file not found at $REPO_ROOT/.env"
    print_info "Please run ./setup.sh first to generate configuration"
    exit 1
fi

# Source the .env file
source "$REPO_ROOT/.env"

# Verify variables
if [ -z "$LLM_ROOT_PATH" ]; then
    print_error "LLM_ROOT_PATH not set in .env"
    exit 1
fi

print_header "DGX Spark LLM Stack - Model Download"
echo "Loading configuration from: $REPO_ROOT/.env"
echo "Storage path: $LLM_ROOT_PATH"
echo ""

# Create model directories
mkdir -p "$LLM_ROOT_PATH"/{vllm,ollama,gguf}

###############################################################################
# CHECK DEPENDENCIES
###############################################################################

print_header "Checking Dependencies"

if ! command -v hf &> /dev/null; then
    print_error "hf command not found"
    print_info "Install with: pip install huggingface-hub[cli]"
    exit 1
fi

print_success "hf command available"

# Optional: Check for hf_transfer for faster downloads
if command -v python3 &> /dev/null; then
    if python3 -c "import hf_transfer" 2>/dev/null; then
        print_success "hf_transfer installed (downloads will be faster)"
        export HF_HUB_ENABLE_HF_TRANSFER=1
    else
        print_info "Optional: pip install hf-transfer for faster downloads"
    fi
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN not set in .env"
    print_info "Some models may be rate-limited without a token"
    echo "To set it: export HF_TOKEN=hf_..."
fi

###############################################################################
# S TIER MODELS (4B-30B)
###############################################################################

if [ "$INSTALL_S_TIER" = "true" ]; then
    print_header "Downloading S Tier Models (4B-30B)"
    
    MODELS=(
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
        "unsloth/Qwen2-7B"
        "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
        "HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive"
    )
    
    for model in "${MODELS[@]}"; do
        print_info "Downloading $model..."
        if hf download "$model" --repo-type model --local-dir "$LLM_ROOT_PATH/vllm/$model" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model (may already exist)"
        fi
    done
else
    print_info "S tier skipped (INSTALL_S_TIER=false)"
fi

###############################################################################
# M TIER MODELS (30B-35B)
###############################################################################

if [ "$INSTALL_M_TIER" = "true" ]; then
    print_header "Downloading M Tier Models (30B-35B)"
    
    MODELS=(
        "microsoft/Phi-4"
        "meta-llama/Llama-3.1-34B-Instruct"
    )
    
    for model in "${MODELS[@]}"; do
        print_info "Downloading $model..."
        if hf download "$model" --repo-type model --local-dir "$LLM_ROOT_PATH/vllm/$model" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model (may already exist or requires approval)"
        fi
    done
else
    print_info "M tier skipped (INSTALL_M_TIER=false)"
fi

###############################################################################
# L TIER MODELS (120B+)
###############################################################################

if [ "$INSTALL_L_TIER" = "true" ]; then
    print_header "Downloading L Tier Models (120B+)"
    
    print_warning "Large models (120B+) may take significant time and disk space (100+ GB)"
    read -p "Continue downloading large models? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        MODELS=(
            "meta-llama/Llama-3.3-70B-Instruct"
            "meta-llama/Llama-3.1-405B-Instruct"
        )
        
        for model in "${MODELS[@]}"; do
            print_info "Downloading $model..."
            print_warning "This may take 30+ minutes depending on internet speed"
            if hf download "$model" --repo-type model --local-dir "$LLM_ROOT_PATH/vllm/$model" 2>&1 | tail -5; then
                print_success "Downloaded: $model"
            else
                print_warning "Failed to download $model (may require approval or API access)"
            fi
        done
    else
        print_info "Large model download cancelled"
    fi
else
    print_info "L tier skipped (INSTALL_L_TIER=false)"
fi

###############################################################################
# GGUF VARIANTS (llama.cpp MODELS)
###############################################################################

if [ "$INSTALL_GGUF" = "true" ]; then
    print_header "Downloading GGUF Models (llama.cpp)"
    
    print_info "Downloading Q4_K_M quantized variants (optimized for GB10 memory)"
    
    GGUF_MODELS=(
        "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
        "lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF"
        "lmstudio-community/phi-4-GGUF"
    )
    
    for model in "${GGUF_MODELS[@]}"; do
        print_info "Downloading $model (Q4_K_M only)..."
        # Only download Q4_K_M quantization to save disk space (20-30 GB instead of 100+ GB)
        if hf download "$model" --repo-type model --include "*Q4_K_M*" --local-dir "$LLM_ROOT_PATH/gguf/$model" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model"
        fi
    done
else
    print_info "GGUF models skipped (INSTALL_GGUF=false)"
fi

###############################################################################
# SUMMARY
###############################################################################

print_header "Download Complete!"

echo -e "${BLUE}Model Summary:${NC}\n"
echo "VLLMs:    $(ls -d $LLM_ROOT_PATH/vllm/*/ 2>/dev/null | wc -l) models"
echo "GGUFs:    $(ls -d $LLM_ROOT_PATH/gguf/*/ 2>/dev/null | wc -l) models"
echo ""

# Calculate disk usage
TOTAL_SIZE=$(du -sh "$LLM_ROOT_PATH" 2>/dev/null | cut -f1)
echo -e "${BLUE}Total Disk Usage:${NC} $TOTAL_SIZE\n"

print_info "Models are ready to use!"
print_info "Next steps:"
echo "  1. Start the stack: docker compose up -d"
echo "  2. Monitor services: docker compose logs -f"
echo "  3. Test the API: curl http://localhost:14000/v1/models"
echo ""

print_success "Happy model serving! 🚀\n"
