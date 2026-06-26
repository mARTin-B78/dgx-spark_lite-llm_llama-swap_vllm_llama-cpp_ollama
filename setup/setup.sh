#!/bin/bash

###############################################################################
# DGX Spark LLM Stack - Interactive Setup Script
# 
# This script guides you through setting up the complete LLM stack with:
# - Docker configuration
# - Service ports (Portainer, LiteLLM, llama-swap, llama.cpp, Ollama, vLLM)
# - Model selection
# - API credentials
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

ask_yes_no() {
    local prompt="$1"
    local response
    while true; do
        read -rp "$(echo -e ${BLUE}${prompt}${NC} ' (y/n): ')" response </dev/tty
        case "${response}" in
            [yY][eE][sS]|[yY]) return 0 ;;
            [nN][oO]|[nN]) return 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

ask_input() {
    local prompt="$1"
    local default="$2"
    local response
    # Read from /dev/tty so input works even when this function is invoked
    # inside command substitution (NAME=$(ask_input ...)). Without this,
    # some shells/terminals drop the keystrokes and the default is silently used.
    read -rp "$(echo -e ${BLUE}$prompt${NC}) [$default]: " response </dev/tty
    echo "${response:-$default}"
}

check_port_in_use() {
    local port=$1
    if nc -z localhost "${port}" 2>/dev/null; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

get_port() {
    local QUESTION="${1}"
    local REQUESTED_PORT="${2}"

    # Everything inside these braces displays to the user's monitor (stderr)
    {
        REQUESTED_PORT=$(ask_input "${QUESTION}" "${REQUESTED_PORT}")

        while check_port_in_use "${REQUESTED_PORT}"; do
            print_warning "Port ${REQUESTED_PORT} is already in use"

            # Automatic free port lookup
            local SUGGESTED_PORT=$((REQUESTED_PORT + 1))
            while check_port_in_use "${SUGGESTED_PORT}"; do
                SUGGESTED_PORT=$((SUGGESTED_PORT + 1))
            done

            REQUESTED_PORT=$(ask_input "Enter a different port" "${SUGGESTED_PORT}")
        done
    } >&2

    # Only this clean numeric value is passed back to the main script variable
    echo "${REQUESTED_PORT}"
}

check_container_running() {
    local container_name=$1
    if ${DOCKER} ps --filter "name=${container_name}" --format '{{.Names}}' 2>/dev/null | grep -q "${container_name}"; then
        return 0 # Container is running
    else
        return 1 # Container is not running
    fi
}

ask_yes_no_default() {
    local prompt="$1"
    local current="$2"
    local default_hint reply
    default_hint=$([[ "${current}" = "true" ]] && echo "Y/n" || echo "y/N")
    read -rp "$(echo -e "${BLUE}${prompt}? (${default_hint}): ${NC}")" -n 1 reply </dev/tty
    echo >&2  # newline after single-char read; to stderr so it isn't captured
    [[ -z "${reply}" ]] && reply=$([[ "${current}" = "true" ]] && echo "y" || echo "n")
    [[ "${reply}" =~ ^[Yy]$ ]] && echo "true" || echo "false"
}

###############################################################################
# SAFETY CHECKS
###############################################################################

if [ "${EUID}" -eq 0 ]; then
    print_error "Do not run this script as root or with sudo — generated files would be owned by root."
    print_info "If docker commands fail, add your user to the docker group and re-login:"
    print_info "  sudo usermod -aG docker \$USER"
    exit 1
fi

###############################################################################
# Load existing .env as defaults (re-run friendly)
###############################################################################

REPO_CONFIG_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "${REPO_CONFIG_PATH}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_CONFIG_PATH}/.env"
    set +a
    print_info "Loaded existing .env from ${REPO_CONFIG_PATH} — existing values will be used as defaults."
fi

LLM_ROOT_PATH="${LLM_ROOT_PATH:-${HOME}/LLMs}"

POSTGRES_DB=${POSTGRES_DB:-litellm}
POSTGRES_USER=${POSTGRES_USER:-litellm_admin}
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-litellm-password-123}"
LITELLM_UI_USERNAME="${LITELLM_UI_USERNAME:-admin}"
LITELLM_UI_PASSWORD="${LITELLM_UI_PASSWORD:-choose-a-ui-password}"
# we do not want to lose the key when it's set in the .env file
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}"

LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-18080}"
LLAMA_QWEN35_4B_PORT="${LLAMA_QWEN35_4B_PORT:-19001}"
LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-28080}"
LITELLM_PORT="${LITELLM_PORT:-4000}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
PORTAINER_PORT="${PORTAINER_PORT:-9443}"
VLLM_PORT="${VLLM_PORT:-18000}"

EXTERNAL_NETWORK_NAME="${EXTERNAL_NETWORK_NAME:-dgx_net}"

REGISTRY_USER="${REGISTRY_USER:-${GH_USER:-your-registry-username}}"
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE}" # will default to :-${REGISTRY_USER} once REGISTRY_USER is set

# User generated tokens cannot have default values
REGISTRY_TOKEN="${REGISTRY_TOKEN:-${GITHUB_TOKEN:-}}"
HF_TOKEN="${HF_TOKEN:-}"

# Derive per-service install defaults from COMPOSE_PROFILES (re-run friendly)
_profile_active() { echo ",${COMPOSE_PROFILES}," | grep -q ",${1},"; }
LITELLM_IN_PROFILES=$(  _profile_active "litellm"      && echo "true" || echo "false")
LLAMASWAP_IN_PROFILES=$(_profile_active "llama-swap"   && echo "true" || echo "false")
LLAMACPP_IN_PROFILES=$( _profile_active "llama-server" && echo "true" || echo "false")
OLLAMA_IN_PROFILES=$(   _profile_active "ollama"       && echo "true" || echo "false")
VLLM_IN_PROFILES=$(     _profile_active "vllm"         && echo "true" || echo "false")

###############################################################################
# START OF SETUP
###############################################################################

print_header "DGX Spark LLM Stack Setup"
echo "This script will guide you through configuring your AI orchestration stack."
echo "It will check for existing services and help you configure ports and credentials."

read -p "Press Enter to continue..." </dev/tty

###############################################################################
# 1. DOCKER CHECK
###############################################################################

print_header "Step 1: Docker & NVIDIA Runtime"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker is installed: ${DOCKER_VERSION}"
else
    print_warning "Docker is not installed"
    print_info "Docker is a containerization platform that packages applications into isolated environments. It's essential for running multiple LLM services simultaneously without conflicts."
    if ask_yes_no "Would you like to install Docker?"; then
        print_info "Visit: https://docs.docker.com/engine/install/ubuntu/"
        read -p "Press Enter after installing Docker..." </dev/tty
    else
        print_error "Docker is required for this setup. Cannot continue."
        exit 1
    fi
fi

if docker ps >/dev/null 2>&1; then
    DOCKER="docker"
else
    print_warning "Cannot run docker without sudo (user not in the docker group)."
    print_info "To fix permanently: sudo usermod -aG docker \$USER  (then log out and back in)"
    DOCKER="sudo docker"
fi


# Check NVIDIA Container Runtime
# Ask Docker directly instead of pulling a test image — `nvidia/cuda:11.0-runtime`
# has no arm64 variant and is too old for Blackwell, so it would falsely fail on the GB10.
if ${DOCKER} info 2>/dev/null | grep -qE '^\s*Default Runtime:\s*nvidia\b'; then
    print_success "NVIDIA Container Runtime is configured (default runtime: nvidia)"
elif ${DOCKER} info 2>/dev/null | grep -qE '^\s*Runtimes:.*\bnvidia\b'; then
    print_warning "NVIDIA runtime is registered but is not the default runtime"
    print_info "Run the following to make it the default, then restart Docker:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
    echo "    sudo systemctl restart docker"
elif command -v nvidia-ctk &>/dev/null; then
    print_warning "NVIDIA Container Runtime not configured"
    print_info "The toolkit is installed but the runtime is not registered with Docker."
    print_info "Run the following to configure it as the default runtime, then restart Docker:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
    echo "    sudo systemctl restart docker"
else
    print_warning "NVIDIA Container Toolkit is not installed"
    print_info "This is required to give containers GPU access on the GB10."
    print_info "Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    print_info "Then run:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
    echo "    sudo systemctl restart docker"
fi

###############################################################################
# 2. DOCKER NETWORK
###############################################################################

print_header "Step 2: Docker Network"

NETWORK_NAME=$(ask_input "Docker network name for the stack" "${EXTERNAL_NETWORK_NAME}")

if ${DOCKER} network ls --filter name="${NETWORK_NAME}" --format '{{.Name}}' 2>/dev/null | grep -q "${NETWORK_NAME}"; then
    print_success "Network '${NETWORK_NAME}' already exists"
else
    print_info "Creating network '${NETWORK_NAME}'..."
    ${DOCKER} network create "${NETWORK_NAME}"
    print_success "Network created"
fi

###############################################################################
# 3. PORTAINER
###############################################################################

print_header "Step 3: Portainer (Optional Container Management UI)"

print_info "Portainer provides a web interface to manage Docker containers, images, and networks."
print_info "It's useful for monitoring the stack, viewing logs, and managing services visually."

if check_container_running "portainer"; then
    print_success "Portainer is already running"
    PORTAINER_CONFIGURED=true
elif ask_yes_no "Install Portainer?"; then
    PORTAINER_PORT=$(get_port "Portainer port (HTTPS UI)" "${PORTAINER_PORT}")

    print_info "Pulling and starting portainer/portainer-ce:latest on port ${PORTAINER_PORT}..."
    if ${DOCKER} volume inspect portainer_data >/dev/null 2>&1; then
        print_info "Reusing existing portainer_data volume"
    else
        ${DOCKER} volume create portainer_data >/dev/null
    fi

    if ${DOCKER} run -d \
        --name portainer \
        --restart=always \
        -p "${PORTAINER_PORT}:9443" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v portainer_data:/data \
        portainer/portainer-ce:latest >/dev/null; then
        print_success "Portainer started: https://localhost:${PORTAINER_PORT}"
        PORTAINER_CONFIGURED=true
    else
        print_error "Failed to start Portainer container (see docker output above)"
        PORTAINER_CONFIGURED=false
    fi
else
    print_info "Skipping Portainer installation"
    PORTAINER_CONFIGURED=false
fi

###############################################################################
# 4. LITELLM
###############################################################################

print_header "Step 4: LiteLLM (API Gateway)"

print_info "LiteLLM is a unified API gateway that provides a single OpenAI-compatible endpoint."
print_info "It routes requests to the appropriate model container and handles authentication."

LITELLM_CONFIGURED=false
if check_container_running "litellm"; then
    print_success "LiteLLM is already running"
    LITELLM_CONFIGURED=true
elif [[ "$(ask_yes_no_default "Install LiteLLM" "${LITELLM_IN_PROFILES}")" = "true" ]]; then
    LITELLM_PORT=$(get_port "LiteLLM API port" "${LITELLM_PORT}")

    print_success "Will use port ${LITELLM_PORT} for LiteLLM"
    LITELLM_UI_USERNAME=$(ask_input "LiteLLM web UI username" "${LITELLM_UI_USERNAME}")
    LITELLM_UI_PASSWORD=$(ask_input "LiteLLM web UI password" "${LITELLM_UI_PASSWORD}")
    POSTGRES_DB=$(ask_input "LiteLLM PostgreSQL database" "${POSTGRES_DB}")
    POSTGRES_USER=$(ask_input "LiteLLM PostgreSQL username" "${POSTGRES_USER}")
    POSTGRES_PASSWORD=$(ask_input "LiteLLM PostgreSQL password" "${POSTGRES_PASSWORD}")
    LITELLM_CONFIGURED=true
fi

###############################################################################
# 5. LLAMA-SWAP
###############################################################################

print_header "Step 5: llama-swap (VRAM Orchestrator)"

print_info "llama-swap manages VRAM allocation, automatically loading/unloading models on demand."
print_info "This is the core of the stack — it enables seamless multi-model switching on 128GB."

LLAMASWAP_CONFIGURED=false
if check_container_running "llama-swap"; then
    print_success "llama-swap is already running"
    LLAMASWAP_CONFIGURED=true
elif [[ "$(ask_yes_no_default "Install llama-swap" "${LLAMASWAP_IN_PROFILES}")" = "true" ]]; then
    LLAMA_SWAP_PORT=$(get_port "llama-swap proxy port" "${LLAMA_SWAP_PORT}")
    
    print_success "Will use port ${LLAMA_SWAP_PORT} for llama-swap"
    LLAMASWAP_CONFIGURED=true
fi

###############################################################################
# 6. LLAMA.CPP
###############################################################################

print_header "Step 6: llama.cpp (GGUF Engine)"

print_info "llama.cpp is an optimized inference engine for quantized GGUF models."
print_info "It provides excellent performance for smaller models (4B-35B) on the GB10."
print_warning "llama-server is disabled by default in docker-compose.yml — llama-swap already spawns ephemeral llama.cpp containers on demand."
print_info "Enable it only if you want a model permanently hot in VRAM. This permanently consumes VRAM that llama-swap cannot reclaim, reducing headroom for other models."

LLAMACPP_CONFIGURED=false
if check_container_running "llama.cpp"; then
    print_success "llama.cpp is already running"
    LLAMACPP_CONFIGURED=true
elif [[ "$(ask_yes_no_default "Install llama.cpp" "${LLAMACPP_IN_PROFILES}")" = "true" ]]; then
    LLAMA_CPP_PORT=$(get_port "llama.cpp API port" "${LLAMA_CPP_PORT}")
    
    print_success "Will use port ${LLAMA_CPP_PORT} for llama.cpp"
    LLAMACPP_CONFIGURED=true
fi

###############################################################################
# 7. OLLAMA
###############################################################################

print_header "Step 7: Ollama (Model Manager)"

print_info "Ollama simplifies running LLMs with built-in model management and caching."
print_info "It's useful for testing models quickly and provides a standalone API."
print_warning "ollama is disabled by default in docker-compose.yml — it runs outside llama-swap's spawn/evict lifecycle."
print_info "Enable it only if you specifically need Ollama's modelfile system. It permanently consumes VRAM that llama-swap cannot reclaim, reducing headroom for other models."

OLLAMA_CONFIGURED=false
if check_container_running "ollama"; then
    print_success "Ollama is already running"
    OLLAMA_CONFIGURED=true
elif [[ "$(ask_yes_no_default "Install Ollama" "${OLLAMA_IN_PROFILES}")" = "true" ]]; then
    OLLAMA_PORT=$(get_port "Ollama API port" "${OLLAMA_PORT}")
    
    print_success "Will use port ${OLLAMA_PORT} for Ollama"
    OLLAMA_CONFIGURED=true
fi

###############################################################################
# 8. VLLM
###############################################################################

print_header "Step 8: vLLM (Persistent Safetensors Engine)"

print_info "vLLM is a high-throughput inference engine for Safetensors (FP8/NVFP4) models."
print_info "It provides excellent performance for large models on the GB10."
print_warning "vllm is disabled by default in docker-compose.yml — llama-swap already spawns ephemeral vLLM containers on demand."
print_info "Enable it only if you want a vLLM sidecar permanently loaded in VRAM. This permanently consumes VRAM that llama-swap cannot reclaim, reducing headroom for other models."

VLLM_CONFIGURED=false
if check_container_running "vllm"; then
    print_success "vLLM is already running"
    VLLM_CONFIGURED=true
elif [[ "$(ask_yes_no_default "Install vLLM (persistent sidecar)" "${VLLM_IN_PROFILES}")" = "true" ]]; then
    VLLM_PORT=$(get_port "vLLM API port" "${VLLM_PORT}")

    print_success "Will use port ${VLLM_PORT} for vLLM"
    VLLM_CONFIGURED=true
fi


###############################################################################
# 9. PATHS & STORAGE
###############################################################################

print_header "Step 9: Storage Paths"

LLM_ROOT_PATH=$(ask_input "Path where model files will be stored" "${LLM_ROOT_PATH}")

print_info "LLM storage: ${LLM_ROOT_PATH}"
print_info "Repo config: ${REPO_CONFIG_PATH}"

# Create directories if they don't exist
mkdir -p "${LLM_ROOT_PATH}"/{vllm,ollama}
mkdir -p "${REPO_CONFIG_PATH}/llama-swap/scripts"

print_success "Directories created/verified"

###############################################################################
# 10. GITHUB/HUGGINGFACE CREDENTIALS
###############################################################################

print_header "Step 10: Model Repository Credentials"

print_info "HuggingFace token is needed to download models from private or rate-limited repos."
HF_TOKEN=$(ask_input "HuggingFace User Access Token (or leave blank to skip)" "${HF_TOKEN}")

print_info "Container registry credentials (used to push the stack images)."
print_info "Default is GitHub Container Registry (ghcr.io). For GitLab/Harbor/Nexus,"
print_info "set REGISTRY and IMAGE_NAMESPACE accordingly."
REGISTRY=$(ask_input "Container registry hostname" "${REGISTRY}")
REGISTRY_USER=$(ask_input "Registry login user (GitHub username for ghcr.io, robot account for others)" "${REGISTRY_USER}")
IMAGE_NAMESPACE=$(ask_input "Image namespace (path under the registry; for ghcr.io = your username; for GitLab = group/project)" "${IMAGE_NAMESPACE:-${REGISTRY_USER}}")
REGISTRY_TOKEN=$(ask_input "Registry token / password (GitHub PAT with 'write:packages' for ghcr.io, GitLab deploy token, …)" "${REGISTRY_TOKEN}")

###############################################################################
# 11. GENERATE CONFIGURATION FILES
###############################################################################

print_header "Step 11: Generating Configuration Files"

# Generate .env file
print_info "Generating .env file..."

# Preserve profiles not managed by the wizard so a re-run doesn't silently drop them.
_had_qwen35_4b=$(echo ",${COMPOSE_PROFILES}," | grep -q ",llama-qwen35-4b," && echo "true" || echo "false")

# Build COMPOSE_PROFILES from wizard selections
COMPOSE_PROFILES=""
[ "${LITELLM_CONFIGURED}"   = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}litellm"
[ "${LLAMASWAP_CONFIGURED}" = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}llama-swap"
[ "${LLAMACPP_CONFIGURED}"  = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}llama-server"
[ "${OLLAMA_CONFIGURED}"    = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}ollama"
[ "${VLLM_CONFIGURED}"      = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}vllm"
[ "${_had_qwen35_4b}"       = "true" ] && COMPOSE_PROFILES="${COMPOSE_PROFILES:+${COMPOSE_PROFILES},}llama-qwen35-4b"

cat > "${REPO_CONFIG_PATH}/.env" << EOF
# Auto-generated by setup.sh on $(date)
#
# Comma-separated Docker Compose profiles to activate — controls which services start.
# Available profiles: litellm, llama-swap, llama-server, ollama, vllm
# Generated automatically by setup/setup.sh; edit manually to enable/disable services.
# Example: COMPOSE_PROFILES=litellm,llama-swap
COMPOSE_PROFILES=${COMPOSE_PROFILES}

# --- DOCKER REGISTRY ---
REGISTRY=${REGISTRY}
# Login user for the registry
REGISTRY_USER=${REGISTRY_USER}
# Path under the registry that prefixes every image name
# For ghcr.io: same as REGISTRY_USER; for GitLab: group/project
IMAGE_NAMESPACE=${IMAGE_NAMESPACE}
IMAGE_TAG=latest

# --- DOCKER REGISTRY AUTH ---
# Token / password for docker login (GitHub PAT with 'write:packages' for ghcr.io,
# GitLab deploy token, etc.)
REGISTRY_TOKEN=${REGISTRY_TOKEN}

# --- HUGGINGFACE AUTH ---
HF_TOKEN=${HF_TOKEN}

# --- PATHS ---
# Root folder for all model files
LLM_ROOT_PATH=${LLM_ROOT_PATH}
# Local path to this repository clone
REPO_CONFIG_PATH=${REPO_CONFIG_PATH}

# LiteLLM
LITELLM_UI_USERNAME=${LITELLM_UI_USERNAME}
LITELLM_UI_PASSWORD=${LITELLM_UI_PASSWORD}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
# optional: enforce API key auth; omit for open API / UI-only auth
#           uncomment the LITELLM_MASTER_KEY environment variable in
#           docker-compose.yml to enable the key
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}

# --- HARDWARE & BUILD ---
# SM121 is for Blackwell GB10 (used by build_and_push.sh)
CUDA_ARCH=121

# Network
EXTERNAL_NETWORK_NAME=${NETWORK_NAME}

# Ports
LITELLM_PORT=${LITELLM_PORT}
LLAMA_SWAP_PORT=${LLAMA_SWAP_PORT}
LLAMA_CPP_PORT=${LLAMA_CPP_PORT}
LLAMA_QWEN35_4B_PORT=${LLAMA_QWEN35_4B_PORT}
OLLAMA_PORT=${OLLAMA_PORT}
VLLM_PORT=${VLLM_PORT}
PORTAINER_PORT=${PORTAINER_PORT}
EOF
print_success ".env file generated"

# Generate docker-compose.yml from sample
print_info "Generating docker-compose.yml..."
if [ -f "${REPO_CONFIG_PATH}/docker-compose.yml.sample" ]; then
    if [ -f "${REPO_CONFIG_PATH}/docker-compose.yml" ] && ! ask_yes_no "docker-compose.yml already exists. Overwrite it?"; then
        print_info "Keeping the existing docker-compose.yml file."
    else
        cp "${REPO_CONFIG_PATH}/docker-compose.yml.sample" "${REPO_CONFIG_PATH}/docker-compose.yml"
        print_success "docker-compose.yml generated"
    fi
else
    print_error "docker-compose.yml.sample not found. Skipping..."
fi


###############################################################################
# 12. SUMMARY
###############################################################################

print_header "Setup Complete!"

echo -e "${GREEN}Configuration Summary:${NC}\n"
echo "Network:          ${NETWORK_NAME}"
echo "Storage Path:     ${LLM_ROOT_PATH}"
echo "Repo Path:        ${REPO_CONFIG_PATH}"
echo ""
echo -e "${BLUE}Services:${NC}"
[ "${PORTAINER_CONFIGURED}" = "true" ] && echo "  ✅ Portainer:    https://localhost:${PORTAINER_PORT}"
[ "${LITELLM_CONFIGURED}" = "true" ] && echo "  ✅ LiteLLM:      http://localhost:${LITELLM_PORT}"
[ "${LLAMASWAP_CONFIGURED}" = "true" ] && echo "  ✅ llama-swap:   http://localhost:${LLAMA_SWAP_PORT}"
[ "${LLAMACPP_CONFIGURED}" = "true" ] && echo "  ✅ llama.cpp:    http://localhost:${LLAMA_CPP_PORT}"
[ "${OLLAMA_CONFIGURED}" = "true" ] && echo "  ✅ Ollama:       http://localhost:${OLLAMA_PORT}"
[ "${VLLM_CONFIGURED}" = "true" ] && echo "  ✅ vLLM:         http://localhost:${VLLM_PORT}"
echo ""

print_info "Configuration files saved:"
echo "  - .env (credentials and paths)"
echo "  - docker-compose.yml (service definitions)"
echo "  - llama-swap/config.yaml.sample (model configuration)"
echo ""

print_header "Next Steps"

echo -e "${YELLOW}1. Review the generated files before deploying:${NC}"
echo "   cat .env"
echo "   cat docker-compose.yml"
echo ""

echo -e "${YELLOW}2. Download models (if you selected model tiers):${NC}"
echo "   ./setup/download-models.sh  # (Optional companion script)"
echo ""

echo -e "${YELLOW}3. Build and push the stack images to the registry:${NC}"
echo "   ./build_and_push.sh"
echo ""

echo -e "${YELLOW}4. (vLLM models only, one-time, ~30-60 min) Build ephemeral inference images:${NC}"
echo "   git submodule update --init --recursive"
echo "   cd vllm/build/spark-vllm-docker"
echo "   ./build-and-copy.sh              # vllm-node:latest        (most models)"
echo "   ./build-and-copy.sh --tf5        # vllm-node-tf5:latest    (Mamba/hybrid models)"
echo "   ./build-and-copy.sh --exp-mxfp4  # vllm-node-mxfp4:latest  (GPT-OSS-120B, optional)"
echo "   cd ../../.."
echo ""

echo -e "${YELLOW}5. Authenticate to the container registry (required to pull images):${NC}"
echo "   docker login ${REGISTRY} -u ${REGISTRY_USER}"
echo "   # Docker will prompt for your password / personal access token"
echo "   # (Skip if you just ran step 3 — build_and_push.sh already logged in)"
echo ""

echo -e "${YELLOW}6. Start the stack:${NC}"
print_info "COMPOSE_PROFILES is set to: '${COMPOSE_PROFILES}' — only those services will start."
if [ "${LLAMACPP_CONFIGURED}" = "true" ] || [ "${VLLM_CONFIGURED}" = "true" ] || [ "${OLLAMA_CONFIGURED}" = "true" ]; then
    print_warning "OOM risk: the persistent inference service(s) you selected hold VRAM permanently."
    print_info "llama-swap cannot reclaim this VRAM. On a 128 GB GB10 this reduces headroom for large models (120B+)."
    echo ""
fi
echo "   docker compose up -d"
echo ""

echo -e "${YELLOW}7. Verify the stack is running:${NC}"
echo "   docker compose ps"
echo "   curl http://localhost:${LLAMA_SWAP_PORT}/v1/models"
echo ""

echo -e "${GREEN}Happy stacking! 🚀${NC}\n"


# Save configuration to a summary file
cat > "${REPO_CONFIG_PATH}/setup/SETUP_SUMMARY.txt" << EOF
DGX Spark LLM Stack - Setup Summary
Generated: $(date)

PATHS:
  LLM Storage:  ${LLM_ROOT_PATH}
  Repo Config:  ${REPO_CONFIG_PATH}

NETWORK:
  Name: ${NETWORK_NAME}

SERVICES & PORTS:
  Portainer:    $([[ "${PORTAINER_CONFIGURED}" = "true" ]] && echo "localhost:${PORTAINER_PORT}" || echo "Not installed")
  LiteLLM:      $([[ "${LITELLM_CONFIGURED}" = "true" ]] && echo "localhost:${LITELLM_PORT}" || echo "Not installed")
  llama-swap:   $([[ "${LLAMASWAP_CONFIGURED}" = "true" ]] && echo "localhost:${LLAMA_SWAP_PORT}" || echo "Not installed")
  llama.cpp:    $([[ "${LLAMACPP_CONFIGURED}" = "true" ]] && echo "localhost:${LLAMA_CPP_PORT}" || echo "Not installed")
  Ollama:       $([[ "${OLLAMA_CONFIGURED}" = "true" ]] && echo "localhost:${OLLAMA_PORT}" || echo "Not installed")
  vLLM:         $([[ "${VLLM_CONFIGURED}" = "true" ]] && echo "localhost:${VLLM_PORT}" || echo "Not installed")

To view this again: cat ${REPO_CONFIG_PATH}/setup/SETUP_SUMMARY.txt
EOF

print_success "Setup summary saved to setup/SETUP_SUMMARY.txt"
