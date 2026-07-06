#!/bin/bash
set -e

echo "Setting up AEON-7 DFlash and PrismaQuant models..."

# 1. Pull the DFlash Docker Image
echo "Pulling DFlash container..."
docker pull ghcr.io/aeon-7/vllm-dflash:latest

# 2. Download the models (using hf_transfer for speed)
export HF_HUB_ENABLE_HF_TRANSFER=1
LLM_ROOT="${LLM_ROOT_PATH:-/home/sparky/LLMs}"
mkdir -p "$LLM_ROOT/vllm/AEON-7"
mkdir -p "$LLM_ROOT/vllm/rdtand"
mkdir -p "$LLM_ROOT/vllm/z-lab"

echo "Downloading AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4..."
hf download AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4 --local-dir "$LLM_ROOT/vllm/AEON-7/DFlash-Qwen3.5-27B-Uncensored-NVFP4" || true

echo "Downloading DFlash Drafter z-lab/Qwen3.5-27B-DFlash..."
hf download z-lab/Qwen3.5-27B-DFlash --local-dir "$LLM_ROOT/vllm/z-lab/Qwen3.5-27B-DFlash" || true

echo "Downloading rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm..."
hf download rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm --local-dir "$LLM_ROOT/vllm/rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm" || true

echo "Models downloaded successfully."
