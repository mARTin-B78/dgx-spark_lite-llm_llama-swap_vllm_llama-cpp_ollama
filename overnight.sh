#!/bin/bash
cd /home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

MODELS="
DeepSeek-V4-Flash-IQ2XXS-DS4
GPT-OSS-120B
Nemotron-3-Nano-4B-FP8
Nemotron-3-Nano-30B-A3B-NVFP4
Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
Nemotron-3-Super-120B-A12B-NVFP4
Qwen3-Coder-Next-FP8-Dynamic
Qwen3-Coder-Next-int4-AutoRound
Qwen3-Omni-30B-A3B-Instruct
Qwen3-VL-30B-A3B-Instruct-FP8
Qwen3.5-4B-Q4_K_M
Qwen3.5-27B-Uncensored-DFlash-NVFP4
Qwen3.5-35B-A3B-FP8
Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M
Qwen3.5-122B-A10B-int4-AutoRound
Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP
Qwen3.6-27B-PrismaQuant-5.5bit
Qwen3.6-27B-PrismaSCOUT-NVFP4
Qwen3.6-27B-uncensored-heretic-vllm
Qwen3.6-35B-A3B-FP8
"

# Run the full benchmark suite which covers load times, memory impact, 
# generation speed, context lengths, and runs the full 69-scenario quality 
# evaluation covering coding, agentic tool use, and creative writing.
for model in $MODELS; do
    echo "=================================================="
    echo "Benchmarking $model..."
    echo "=================================================="
    ./benchmark-models.sh --quality --quality-mode full "$model"
done
