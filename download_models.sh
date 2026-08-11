#!/bin/bash
set -e

echo "Downloading Abliterated model..."
# DISABLING hf_transfer to prevent stalling on part 11
export HF_HUB_ENABLE_HF_TRANSFER=0
LLM_ROOT="/home/sparky/LLMs"
mkdir -p "$LLM_ROOT/vllm/catplusplus"
mkdir -p "$LLM_ROOT/vllm/Alibaba"
mkdir -p "$LLM_ROOT/vllm/RedHatAI"
mkdir -p "$LLM_ROOT/vllm/sjug"
mkdir -p "$LLM_ROOT/vllm/rdtand"

# Run download
huggingface-cli download Intel/Qwen3.5-122B-A10B-int4-AutoRound --local-dir "$LLM_ROOT/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound" > /tmp/dl_qwen35_autoround.log 2>&1
huggingface-cli download sjug/Qwen3.5-122B-A10B-NVFP4-resharded --local-dir "$LLM_ROOT/vllm/sjug/Qwen3.5-122B-A10B-NVFP4-resharded" > /tmp/dl_qwen35_nvfp4.log 2>&1
huggingface-cli download rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm --local-dir "$LLM_ROOT/vllm/rdtand/Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm" > /tmp/dl_qwen36_prismaquant_35b.log 2>&1

echo "Downloads complete."
