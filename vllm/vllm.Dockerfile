# Optimized vLLM for DGX Spark (GB10)
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS build

# Install ARM64-specific wheels for vLLM & FlashInfer
RUN apt-get update && apt-get install -y python3-pip git curl libgomp1

# Use the eugr/spark-vllm-docker recommendation: 
# Pull the nightly wheels built specifically for Grace-Blackwell
RUN pip install --no-cache-dir vllm --extra-index-url https://wheels.vllm.ai/nightly/

# Runtime Stage
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure FlashInfer and MoE backends are optimized for SM12.1
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV VLLM_FLASHINFER_MOE_BACKEND=latency
ENV GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
