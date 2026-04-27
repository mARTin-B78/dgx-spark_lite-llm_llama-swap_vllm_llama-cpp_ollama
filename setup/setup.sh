#!/bin/bash

###############################################################################
# DGX Spark LLM Stack - Interactive Setup Script
# 
# This script guides you through setting up the complete LLM stack with:
# - Docker configuration
# - Service ports (Portainer, LiteLLM, llama.cpp, Ollama, llama-swap)
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
        read -p "$(echo -e ${BLUE}$prompt${NC} ' (y/n): ')" response
        case "$response" in
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
    read -p "$(echo -e ${BLUE}$prompt${NC}) [$default]: " response
    echo "${response:-$default}"
}

check_port_in_use() {
    local port=$1
    if nc -z localhost "$port" 2>/dev/null; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

check_container_running() {
    local container_name=$1
    if docker ps --filter "name=$container_name" --format '{{.Names}}' 2>/dev/null | grep -q "$container_name"; then
        return 0 # Container is running
    else
        return 1 # Container is not running
    fi
}

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
    print_success "Docker is installed: $DOCKER_VERSION"
    DOCKER_INSTALLED=true
else
    print_warning "Docker is not installed"
    print_info "Docker is a containerization platform that packages applications into isolated environments. It's essential for running multiple LLM services simultaneously without conflicts."
    if ask_yes_no "Would you like to install Docker?"; then
        print_info "Visit: https://docs.docker.com/engine/install/ubuntu/"
        read -p "Press Enter after installing Docker..." </dev/tty
        DOCKER_INSTALLED=true
    else
        print_error "Docker is required for this setup. Cannot continue."
        exit 1
    fi
fi

# Check NVIDIA Container Runtime
# Ask Docker directly instead of pulling a test image — `nvidia/cuda:11.0-runtime`
# has no arm64 variant and is too old for Blackwell, so it would falsely fail on the GB10.
if docker info 2>/dev/null | grep -qE '^\s*Default Runtime:\s*nvidia\b'; then
    print_success "NVIDIA Container Runtime is configured (default runtime: nvidia)"
elif docker info 2>/dev/null | grep -qE '^\s*Runtimes:.*\bnvidia\b'; then
    print_warning "NVIDIA runtime is registered but is not the default runtime"
    print_info "Containers must be started with --runtime=nvidia, or set 'default-runtime': 'nvidia' in /etc/docker/daemon.json."
else
    print_warning "NVIDIA Container Runtime not configured or not available"
    print_info "This is required to give containers GPU access on the GB10."
    echo "Check /etc/docker/daemon.json has:"
    echo '  "default-runtime": "nvidia"'
fi

###############################################################################
# 2. DOCKER NETWORK
###############################################################################

print_header "Step 2: Docker Network"

NETWORK_NAME=$(ask_input "Docker network name for the stack" "dgx_net")

if docker network ls --filter name="$NETWORK_NAME" --format '{{.Name}}' 2>/dev/null | grep -q "$NETWORK_NAME"; then
    print_success "Network '$NETWORK_NAME' already exists"
else
    print_info "Creating network '$NETWORK_NAME'..."
    docker network create "$NETWORK_NAME"
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
    PORTAINER_PORT=9000
    PORTAINER_CONFIGURED=true
elif ask_yes_no "Install Portainer?"; then
    PORTAINER_PORT=$(ask_input "Portainer port" "9000")
    
    # Check if port is in use
    if check_port_in_use "$PORTAINER_PORT"; then
        print_warning "Port $PORTAINER_PORT is already in use"
        PORTAINER_PORT=$(ask_input "Enter a different port" "9001")
    fi
    
    print_success "Will use port $PORTAINER_PORT for Portainer"
    PORTAINER_CONFIGURED=true
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
    LITELLM_PORT=4000
    LITELLM_CONFIGURED=true
elif ask_yes_no "Install LiteLLM?"; then
    LITELLM_PORT=$(ask_input "LiteLLM API port" "4000")
    
    # Check if port is in use
    if check_port_in_use "$LITELLM_PORT"; then
        print_warning "Port $LITELLM_PORT is already in use"
        LITELLM_PORT=$(ask_input "Enter a different port" "4001")
    fi
    
    print_success "Will use port $LITELLM_PORT for LiteLLM"
    LITELLM_MASTER_KEY=$(ask_input "LiteLLM Master API Key (generate one: 'sk-' + random string)" "sk-your-secure-key-here")
    LITELLM_POSTGRES_PASSWORD=$(ask_input "LiteLLM PostgreSQL password" "litellm-password-123")
    LITELLM_CONFIGURED=true
fi

###############################################################################
# 5. LLAMA.CPP
###############################################################################

print_header "Step 5: llama.cpp (GGUF Engine)"

print_info "llama.cpp is an optimized inference engine for quantized GGUF models."
print_info "It provides excellent performance for smaller models (4B-35B) on the GB10."

LLAMACPP_CONFIGURED=false
if check_container_running "llama.cpp"; then
    print_success "llama.cpp is already running"
    LLAMACPP_PORT=18080
    LLAMACPP_CONFIGURED=true
elif ask_yes_no "Install llama.cpp?"; then
    LLAMACPP_PORT=$(ask_input "llama.cpp API port" "18080")
    
    # Check if port is in use
    if check_port_in_use "$LLAMACPP_PORT"; then
        print_warning "Port $LLAMACPP_PORT is already in use"
        LLAMACPP_PORT=$(ask_input "Enter a different port" "18081")
    fi
    
    print_success "Will use port $LLAMACPP_PORT for llama.cpp"
    LLAMACPP_CONFIGURED=true
fi

###############################################################################
# 6. OLLAMA
###############################################################################

print_header "Step 6: Ollama (Model Manager)"

print_info "Ollama simplifies running LLMs with built-in model management and caching."
print_info "It's useful for testing models quickly and provides a standalone API."

OLLAMA_CONFIGURED=false
if check_container_running "ollama"; then
    print_success "Ollama is already running"
    OLLAMA_PORT=11434
    OLLAMA_CONFIGURED=true
elif ask_yes_no "Install Ollama?"; then
    OLLAMA_PORT=$(ask_input "Ollama API port" "11434")
    
    # Check if port is in use
    if check_port_in_use "$OLLAMA_PORT"; then
        print_warning "Port $OLLAMA_PORT is already in use"
        OLLAMA_PORT=$(ask_input "Enter a different port" "11435")
    fi
    
    print_success "Will use port $OLLAMA_PORT for Ollama"
    OLLAMA_CONFIGURED=true
fi

###############################################################################
# 7. LLAMA-SWAP
###############################################################################

print_header "Step 7: llama-swap (VRAM Orchestrator)"

print_info "llama-swap manages VRAM allocation, automatically loading/unloading models on demand."
print_info "This is the core of the stack — it enables seamless multi-model switching on 128GB."

LLAMASWAP_CONFIGURED=false
if check_container_running "llama-swap"; then
    print_success "llama-swap is already running"
    LLAMASWAP_PORT=28080
    LLAMASWAP_CONFIGURED=true
elif ask_yes_no "Install llama-swap?"; then
    LLAMASWAP_PORT=$(ask_input "llama-swap proxy port" "28080")
    
    # Check if port is in use
    if check_port_in_use "$LLAMASWAP_PORT"; then
        print_warning "Port $LLAMASWAP_PORT is already in use"
        LLAMASWAP_PORT=$(ask_input "Enter a different port" "28081")
    fi
    
    print_success "Will use port $LLAMASWAP_PORT for llama-swap"
    LLAMASWAP_CONFIGURED=true
fi

###############################################################################
# 8. PATHS & STORAGE
###############################################################################

print_header "Step 8: Storage Paths"

LLM_ROOT_PATH=$(ask_input "Path where model files will be stored" "$HOME/LLMs")
REPO_CONFIG_PATH=$(pwd)

print_info "LLM storage: $LLM_ROOT_PATH"
print_info "Repo config: $REPO_CONFIG_PATH"

# Create directories if they don't exist
mkdir -p "$LLM_ROOT_PATH"/{vllm,ollama}
mkdir -p "$REPO_CONFIG_PATH/llama-swap/scripts"

print_success "Directories created/verified"

###############################################################################
# 9. GITHUB/HUGGINGFACE CREDENTIALS
###############################################################################

print_header "Step 9: Model Repository Credentials"

print_info "HuggingFace token is needed to download models from private or rate-limited repos."
HF_TOKEN=$(ask_input "HuggingFace User Access Token (or leave blank to skip)" "")

print_info "GitHub PAT is needed to push Docker images to GHCR (optional)."
GH_USER=$(ask_input "GitHub username (or leave blank to skip GHCR)" "")
GH_PAT=$(ask_input "GitHub Personal Access Token with 'write:packages' scope (or leave blank)" "")

###############################################################################
# 10. MODEL SELECTION
###############################################################################

print_header "Step 10: Model Selection"

echo "Select which model tiers to configure:"
echo ""

read -p "$(echo -e "${BLUE}Install S tier (4B-30B small/fast models)? (y/n): ${NC}")" -n 1 -r
echo
INSTALL_S_TIER=$([[ $REPLY =~ ^[Yy]$ ]] && echo "true" || echo "false")

read -p "$(echo -e "${BLUE}Install M tier (30B-35B medium models)? (y/n): ${NC}")" -n 1 -r
echo
INSTALL_M_TIER=$([[ $REPLY =~ ^[Yy]$ ]] && echo "true" || echo "false")

read -p "$(echo -e "${BLUE}Install L tier (120B+ large models)? (y/n): ${NC}")" -n 1 -r
echo
INSTALL_L_TIER=$([[ $REPLY =~ ^[Yy]$ ]] && echo "true" || echo "false")

read -p "$(echo -e "${BLUE}Install GGUF variants (llama.cpp models)? (y/n): ${NC}")" -n 1 -r
echo
INSTALL_GGUF=$([[ $REPLY =~ ^[Yy]$ ]] && echo "true" || echo "false")

###############################################################################
# 11. GENERATE CONFIGURATION FILES
###############################################################################

print_header "Step 11: Generating Configuration Files"

# Generate .env file
print_info "Generating .env file..."
cat > "$REPO_CONFIG_PATH/.env" << EOF
# Auto-generated by setup.sh on $(date)

# Paths
LLM_ROOT_PATH=$LLM_ROOT_PATH
REPO_CONFIG_PATH=$REPO_CONFIG_PATH

# Network
DOCKER_NETWORK=$NETWORK_NAME

# Credentials
GH_USER=${GH_USER:-}
GH_PAT=${GH_PAT:-}
HF_TOKEN=${HF_TOKEN:-}

# Ports
LITELLM_PORT=${LITELLM_PORT:-4000}
LLAMACPP_PORT=${LLAMACPP_PORT:-18080}
OLLAMA_PORT=${OLLAMA_PORT:-11434}
LLAMASWAP_PORT=${LLAMASWAP_PORT:-28080}
PORTAINER_PORT=${PORTAINER_PORT:-9000}

# LiteLLM
LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-sk-your-secure-key}
POSTGRES_PASSWORD=${LITELLM_POSTGRES_PASSWORD:-litellm-password}

# Model tiers
INSTALL_S_TIER=$INSTALL_S_TIER
INSTALL_M_TIER=$INSTALL_M_TIER
INSTALL_L_TIER=$INSTALL_L_TIER
INSTALL_GGUF=$INSTALL_GGUF
EOF
print_success ".env file generated"

# Generate docker-compose.yml from sample
print_info "Generating docker-compose.yml..."
if [ -f "$REPO_CONFIG_PATH/docker-compose.yml.sample" ]; then
    cp "$REPO_CONFIG_PATH/docker-compose.yml.sample" "$REPO_CONFIG_PATH/docker-compose.yml"
    
    # Substitute paths
    sed -i "s|<LLM_ROOT_PATH>|$LLM_ROOT_PATH|g" "$REPO_CONFIG_PATH/docker-compose.yml"
    sed -i "s|<REPO_CONFIG_PATH>|$REPO_CONFIG_PATH|g" "$REPO_CONFIG_PATH/docker-compose.yml"
    sed -i "s|<YOUR_GITHUB_PAT>|${GH_PAT:-}|g" "$REPO_CONFIG_PATH/docker-compose.yml"
    sed -i "s|<YOUR_POSTGRES_PASSWORD>|${LITELLM_POSTGRES_PASSWORD}|g" "$REPO_CONFIG_PATH/docker-compose.yml"
    sed -i "s|<YOUR_LITELLM_MASTER_KEY>|${LITELLM_MASTER_KEY}|g" "$REPO_CONFIG_PATH/docker-compose.yml"
    
    print_success "docker-compose.yml generated"
else
    print_error "docker-compose.yml.sample not found. Skipping..."
fi

###############################################################################
# 12. SUMMARY
###############################################################################

print_header "Setup Complete!"

echo -e "${GREEN}Configuration Summary:${NC}\n"
echo "Network:          $NETWORK_NAME"
echo "Storage Path:     $LLM_ROOT_PATH"
echo "Repo Path:        $REPO_CONFIG_PATH"
echo ""
echo -e "${BLUE}Services:${NC}"
[ "$PORTAINER_CONFIGURED" = "true" ] && echo "  ✅ Portainer:    http://localhost:$PORTAINER_PORT"
[ "$LITELLM_CONFIGURED" = "true" ] && echo "  ✅ LiteLLM:      http://localhost:$LITELLM_PORT"
[ "$LLAMACPP_CONFIGURED" = "true" ] && echo "  ✅ llama.cpp:    http://localhost:$LLAMACPP_PORT"
[ "$OLLAMA_CONFIGURED" = "true" ] && echo "  ✅ Ollama:       http://localhost:$OLLAMA_PORT"
[ "$LLAMASWAP_CONFIGURED" = "true" ] && echo "  ✅ llama-swap:   http://localhost:$LLAMASWAP_PORT"
echo ""
echo -e "${BLUE}Model Tiers:${NC}"
[ "$INSTALL_S_TIER" = "true" ] && echo "  ✅ S tier (4B-30B)"
[ "$INSTALL_M_TIER" = "true" ] && echo "  ✅ M tier (30B-35B)"
[ "$INSTALL_L_TIER" = "true" ] && echo "  ✅ L tier (120B+)"
[ "$INSTALL_GGUF" = "true" ] && echo "  ✅ GGUF variants"
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

echo -e "${YELLOW}3. Start the stack:${NC}"
echo "   docker network create $NETWORK_NAME  # If not already created"
echo "   docker compose up -d"
echo ""

echo -e "${YELLOW}4. Verify the stack is running:${NC}"
echo "   docker compose ps"
echo "   curl http://localhost:$LLAMASWAP_PORT/v1/models"
echo ""

echo -e "${GREEN}Happy stacking! 🚀${NC}\n"

# Save configuration to a summary file
cat > "$REPO_CONFIG_PATH/setup/SETUP_SUMMARY.txt" << EOF
DGX Spark LLM Stack - Setup Summary
Generated: $(date)

PATHS:
  LLM Storage:  $LLM_ROOT_PATH
  Repo Config:  $REPO_CONFIG_PATH

NETWORK:
  Name: $NETWORK_NAME

SERVICES & PORTS:
  Portainer:    $([[ "$PORTAINER_CONFIGURED" = "true" ]] && echo "localhost:$PORTAINER_PORT" || echo "Not installed")
  LiteLLM:      $([[ "$LITELLM_CONFIGURED" = "true" ]] && echo "localhost:$LITELLM_PORT" || echo "Not installed")
  llama.cpp:    $([[ "$LLAMACPP_CONFIGURED" = "true" ]] && echo "localhost:$LLAMACPP_PORT" || echo "Not installed")
  Ollama:       $([[ "$OLLAMA_CONFIGURED" = "true" ]] && echo "localhost:$OLLAMA_PORT" || echo "Not installed")
  llama-swap:   $([[ "$LLAMASWAP_CONFIGURED" = "true" ]] && echo "localhost:$LLAMASWAP_PORT" || echo "Not installed")

MODEL TIERS:
  S tier (4B-30B):    $INSTALL_S_TIER
  M tier (30B-35B):   $INSTALL_M_TIER
  L tier (120B+):     $INSTALL_L_TIER
  GGUF variants:      $INSTALL_GGUF

To view this again: cat $REPO_CONFIG_PATH/setup/SETUP_SUMMARY.txt
EOF

print_success "Setup summary saved to setup/SETUP_SUMMARY.txt"
