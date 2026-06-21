#!/bin/bash

###############################################################################
# DGX Spark LLM Stack - Model Download Script
# 
# This script automatically downloads models based on your tier selections
# from .env. It uses hf download for efficient, resumable downloads.
###############################################################################

set -e
set -o pipefail

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

# Find the repo root (one level up from setup/)
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
    print_warning "hf command not found"
    read -p "Automatically install huggingface-hub[cli] into a local venv? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ! command -v python3 &> /dev/null; then
            print_error "python3 not found — cannot create venv"
            exit 1
        fi
        if ! python3 -c "import venv" 2>/dev/null; then
            print_error "python3-venv is not installed"
            print_info "Install it with: sudo apt install python3-venv"
            exit 1
        fi
        print_info "Creating venv at ~/.local/share/hf-env ..."
        python3 -m venv ~/.local/share/hf-env
        print_info "Installing huggingface-hub ..."
        ~/.local/share/hf-env/bin/pip install -q huggingface-hub
        mkdir -p ~/.local/bin
        ln -sf ~/.local/share/hf-env/bin/hf ~/.local/bin/hf
        export PATH="$HOME/.local/bin:$PATH"
        print_success "hf installed and symlinked to ~/.local/bin/hf (available in all future shells)"
    else
        print_info "To install manually, run:"
        echo "  python3 -m venv ~/.local/share/hf-env"
        echo "  ~/.local/share/hf-env/bin/pip install huggingface-hub"
        echo "  mkdir -p ~/.local/bin"
        echo "  ln -sf ~/.local/share/hf-env/bin/hf ~/.local/bin/hf"
        echo "Then re-open your shell and re-run this script."
        exit 1
    fi
fi

if ! command -v hf &> /dev/null; then
    print_error "hf still not found after install — check your PATH or re-open your shell"
    exit 1
fi

print_success "hf command available"

export HF_XET_HIGH_PERFORMANCE=1
print_success "High-performance Xet transfer enabled"

# Check for HF_TOKEN
HF_AUTH_ARGS=()
if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN not set in .env"
    print_info "Some models may be rate-limited without a token"
    echo "To set it: export HF_TOKEN=hf_..."
else
    # huggingface_hub reads HF_TOKEN from env, but export it explicitly so
    # subprocesses (and the hf CLI) pick it up. Pass --token too as a belt-and-braces.
    export HF_TOKEN
    HF_AUTH_ARGS=(--token "$HF_TOKEN")
    print_success "HF_TOKEN loaded from .env (authenticated downloads)"
fi

###############################################################################
# S TIER MODELS (4B-30B)
###############################################################################

print_header "S Tier Models (4B-30B)"

S_MODELS=(
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    "Intel/Qwen3-Coder-Next-int4-AutoRound"
    "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
    "rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"
)
S_MODELS_DESC=(
    "Nemotron Nano 4B — FP8, small/fast"
    "Nemotron Nano 30B A3B — NVFP4 MoE"
    "Qwen3-Coder-Next — INT4 AutoRound, coding"
    "Qwen3.6 35B A3B — uncensored NVFP4"
    "Qwen3.6 27B — NVFP4/BF16, Blackwell-optimised"
)

echo "Models in this tier:"
for i in "${!S_MODELS[@]}"; do
    printf "  %d. %-55s  %s\n" "$((i+1))" "${S_MODELS[$i]}" "${S_MODELS_DESC[$i]}"
done
echo ""

S_SELECTED=()
echo "Select models to download:"
for i in "${!S_MODELS[@]}"; do
    read -p "  Download ${S_MODELS[$i]}? [y/n]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && S_SELECTED+=("${S_MODELS[$i]}")
done

if [ ${#S_SELECTED[@]} -eq 0 ]; then
    print_info "No S tier models selected"
else
    print_info "Downloading ${#S_SELECTED[@]} of ${#S_MODELS[@]} S tier models..."
    for model in "${S_SELECTED[@]}"; do
        print_info "Downloading $model..."
        if [ "$model" = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" ]; then
            download_cmd=(env HF_HUB_CACHE="$LLM_ROOT_PATH/vllm/nvidia/Nemotron-3-Nano-30B-A3B-NVFP4/hub"
                hf download "$model" --repo-type model
                --revision ab537beb128913d389306a4d0cc1a2097ac5a5af
                "${HF_AUTH_ARGS[@]}")
        else
            download_cmd=(hf download "$model" --repo-type model
                --local-dir "$LLM_ROOT_PATH/vllm/$model"
                "${HF_AUTH_ARGS[@]}")
        fi
        if "${download_cmd[@]}" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model (may already exist)"
        fi
    done
fi

###############################################################################
# M TIER MODELS (30B-35B)
###############################################################################

print_header "M Tier Models (30B-35B)"

M_MODELS=(
    "Qwen/Qwen3.5-35B-A3B-FP8"
    "Qwen/Qwen3.6-35B-A3B-FP8"
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    "unsloth/Qwen3-Coder-Next-FP8-Dynamic"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)
M_MODELS_DESC=(
    "Qwen3.5 35B A3B — FP8 MoE"
    "Qwen3.6 35B A3B — FP8 MoE"
    "Qwen3-VL 30B A3B — vision-language FP8"
    "Qwen3-Omni 30B A3B — audio/image/text FP8"
    "Qwen3-Coder-Next — FP8 Dynamic, coding"
    "Mistral Small 24B — dense FP16"
)

echo "Models in this tier:"
for i in "${!M_MODELS[@]}"; do
    printf "  %d. %-55s  %s\n" "$((i+1))" "${M_MODELS[$i]}" "${M_MODELS_DESC[$i]}"
done
echo ""

M_SELECTED=()
echo "Select models to download:"
for i in "${!M_MODELS[@]}"; do
    read -p "  Download ${M_MODELS[$i]}? [y/n]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && M_SELECTED+=("${M_MODELS[$i]}")
done

if [ ${#M_SELECTED[@]} -eq 0 ]; then
    print_info "No M tier models selected"
else
    print_info "Downloading ${#M_SELECTED[@]} of ${#M_MODELS[@]} M tier models..."
    for model in "${M_SELECTED[@]}"; do
        print_info "Downloading $model..."
        if hf download "$model" --repo-type model --local-dir "$LLM_ROOT_PATH/vllm/$model" "${HF_AUTH_ARGS[@]}" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model (may already exist or requires approval)"
        fi
    done
fi

###############################################################################
# L TIER MODELS (120B+)
###############################################################################

print_header "L Tier Models (120B+)"

print_warning "120B+ models may require 30+ minutes and 60-100 GB of disk space each"

L_MODELS=(
    "Intel/Qwen3.5-122B-A10B-int4-AutoRound"
    "openai/gpt-oss-120b"
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
L_MODELS_DESC=(
    "Qwen3.5 122B A10B — INT4 AutoRound MoE"
    "GPT-OSS 120B — requires HF approval"
    "Nemotron Omni 30B A3B — reasoning FP8"
    "Nemotron Omni 30B A3B — reasoning NVFP4"
    "Nemotron Super 120B A12B — NVFP4 MoE"
)

echo "Models in this tier:"
for i in "${!L_MODELS[@]}"; do
    printf "  %d. %-55s  %s\n" "$((i+1))" "${L_MODELS[$i]}" "${L_MODELS_DESC[$i]}"
done
echo ""

L_SELECTED=()
echo "Select models to download:"
for i in "${!L_MODELS[@]}"; do
    read -p "  Download ${L_MODELS[$i]}? [y/n]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && L_SELECTED+=("${L_MODELS[$i]}")
done

if [ ${#L_SELECTED[@]} -eq 0 ]; then
    print_info "No L tier models selected"
else
    print_info "Downloading ${#L_SELECTED[@]} of ${#L_MODELS[@]} L tier models..."
    for model in "${L_SELECTED[@]}"; do
        print_info "Downloading $model..."
        if [ "$model" = "openai/gpt-oss-120b" ]; then
            download_cmd=(env HF_HUB_CACHE="$LLM_ROOT_PATH/vllm/openai/GPT-OSS-120B/hub"
                hf download "$model" --repo-type model
                --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a
                "${HF_AUTH_ARGS[@]}")
        else
            download_cmd=(hf download "$model" --repo-type model
                --local-dir "$LLM_ROOT_PATH/vllm/$model"
                "${HF_AUTH_ARGS[@]}")
        fi
        if "${download_cmd[@]}" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model (may require approval or API access)"
        fi
    done
fi

###############################################################################
# GGUF VARIANTS (llama.cpp MODELS)
###############################################################################

print_header "GGUF Models (llama.cpp)"

GGUF_MODELS=(
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"
)
GGUF_MODELS_DESC=(
    "Nemotron Nano 4B — Q4_K_M only, optimised for GB10"
)

echo "Models in this tier:"
for i in "${!GGUF_MODELS[@]}"; do
    printf "  %d. %-55s  %s\n" "$((i+1))" "${GGUF_MODELS[$i]}" "${GGUF_MODELS_DESC[$i]}"
done
echo ""

GGUF_SELECTED=()
echo "Select models to download:"
for i in "${!GGUF_MODELS[@]}"; do
    read -p "  Download ${GGUF_MODELS[$i]}? [y/n]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && GGUF_SELECTED+=("${GGUF_MODELS[$i]}")
done

if [ ${#GGUF_SELECTED[@]} -eq 0 ]; then
    print_info "No GGUF models selected"
else
    print_info "Downloading ${#GGUF_SELECTED[@]} of ${#GGUF_MODELS[@]} GGUF models..."
    for model in "${GGUF_SELECTED[@]}"; do
        print_info "Downloading $model (Q4_K_M only)..."
        if hf download "$model" --repo-type model --include "*Q4_K_M*" --local-dir "$LLM_ROOT_PATH/gguf/$model" "${HF_AUTH_ARGS[@]}" 2>&1 | tail -5; then
            print_success "Downloaded: $model"
        else
            print_warning "Failed to download $model"
        fi
    done
fi

###############################################################################
# OLLAMA DIR GGUF MODELS (llama.cpp, stored in ollama/)
###############################################################################

print_header "GGUF Models (ollama dir)"

OLLAMA_REPOS=(
    "llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF"
    "HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive"
    "TheDrummer/GLM-Steam-106B-A12B-v1-GGUF"
)
OLLAMA_INCLUDES=(
    "*NVFP4-MLP-Only*"
    "*BF16*"
    "*Q4_K_M*"
)
OLLAMA_LOCAL_DIRS=(
    "$LLM_ROOT_PATH/ollama/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-NVFP4-GGUF"
    "$LLM_ROOT_PATH/ollama/HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive"
    "$LLM_ROOT_PATH/ollama/TheDrummer/GLM-Steam-106B-A12B-v1-GGUF"
)
OLLAMA_DESCS=(
    "Qwen3.6 27B uncensored heretic-v2 — S tier, NVFP4-MLP-Only"
    "Qwen3.5 27B Uncensored HauhauCS — M tier, BF16"
    "GLM-Steam 106B A12B — L tier, Q4_K_M only (~73 GB)"
)

echo "Models in this tier:"
for i in "${!OLLAMA_REPOS[@]}"; do
    printf "  %d. %-55s  %s\n" "$((i+1))" "${OLLAMA_REPOS[$i]}" "${OLLAMA_DESCS[$i]}"
done
echo ""

OLLAMA_SELECTED_INDICES=()
echo "Select models to download:"
for i in "${!OLLAMA_REPOS[@]}"; do
    read -p "  Download ${OLLAMA_REPOS[$i]}? [y/n]: " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && OLLAMA_SELECTED_INDICES+=("$i")
done

if [ ${#OLLAMA_SELECTED_INDICES[@]} -eq 0 ]; then
    print_info "No Ollama GGUF models selected"
else
    print_info "Downloading ${#OLLAMA_SELECTED_INDICES[@]} of ${#OLLAMA_REPOS[@]} Ollama GGUF models..."
    for i in "${OLLAMA_SELECTED_INDICES[@]}"; do
        print_info "Downloading ${OLLAMA_REPOS[$i]}..."
        if hf download "${OLLAMA_REPOS[$i]}" \
            --repo-type model \
            --include "${OLLAMA_INCLUDES[$i]}" \
            --local-dir "${OLLAMA_LOCAL_DIRS[$i]}" \
            "${HF_AUTH_ARGS[@]}" 2>&1 | tail -5; then
            print_success "Downloaded: ${OLLAMA_REPOS[$i]}"
        else
            print_warning "Failed to download ${OLLAMA_REPOS[$i]}"
        fi
    done
fi

###############################################################################
# SUMMARY
###############################################################################

print_header "Download Complete!"

echo -e "${BLUE}Model Summary:${NC}\n"
echo "vLLM:     $(find "$LLM_ROOT_PATH/vllm" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l) models"
echo "GGUF:     $(find "$LLM_ROOT_PATH/gguf" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l) models"
echo "Ollama:   $(find "$LLM_ROOT_PATH/ollama" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l) models"
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
