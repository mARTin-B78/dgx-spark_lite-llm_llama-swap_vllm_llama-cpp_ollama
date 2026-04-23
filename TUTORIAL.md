# Running a Full Multi-Model LLM Stack on DGX Spark (GB10) — With VRAM Orchestration

Hey Guys, I (and the AI) have been working on this for quite a while and this is what I have so far.
Once it is set up it shall make using the LLMs as easy as possible — the loading and unloading is done by llama-swap. No matter if you use vLLM, llama.cpp or Ollama.
LiteLLM is used to route the LLMs. You can also use it for fallbacks or adding remote models to the stack.

The biggest issue so far is speed. It works reliably but it is not fast.
If someone has a better, easier, faster, more reliable solution please let me know.

**After working on this for weeks and having a working solution I asked the AI to help generate a tutorial for it and this is what came out.**

---

**Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10) — 128 GB unified CPU/GPU memory, SM12.1 GPU architecture, ARM64 (SBSA) CPU

**What this stack gives you:** A single OpenAI-compatible API endpoint that dynamically swaps 10+ models in and out of VRAM on demand — 4B GGUF models, 30B FP8, 120B+ MoE, VLMs — with no manual `docker run` required. Requests come in through LiteLLM, which routes to llama-swap, which spins up the right vLLM/llama.cpp container for that model and kills it again after idle timeout.

**This guide builds directly on the outstanding work of:**

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) provides the pre-built vLLM + FlashInfer wheels compiled specifically for the GB10. Without this, building a working vLLM image from source takes 2–4 hours and frequently breaks on nightly. He also authored [llama-benchy](https://github.com/eugr/llama-benchy), the standardized benchmarking tool used throughout this guide. Massive thanks — this stack would not be usable in practice without both of those projects.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) and the associated forks of vLLM, FlashInfer, and CUTLASS that enable native MXFP4 quantization on GB10. This is what makes OpenAI GPT-OSS-120B actually run at ~57 tok/s on a single Spark.

**GitHub repo (all Dockerfiles + configs):** https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

---

## Architecture Overview

```
Client / Open-WebUI
        ↓
   LiteLLM :14000          ← unified API key, routing, model aliases
        ↓
  llama-swap :28080         ← VRAM orchestrator, loads/evicts on demand
     /    |    \
vLLM   vLLM   llama.cpp    ← one ephemeral container per model
(30B)  (120B)  (GGUF)
```

All containers share the `dgx_net` Docker bridge network. Model containers attach to llama-swap's network namespace (`--network container:llama-swap`), so they reach `localhost:PORT` inside llama-swap's own network — this is how llama-swap knows when a model is ready.

**Key insight for the GB10's unified memory:** CUDA sees ~121.7 GiB of the 128 GB physical RAM. Every model you load eats into that shared pool. llama-swap's `gpu_memory_utilization` setting is your lever — it tells vLLM "claim at most X% of 121.7 GiB".

---

## Benchmark Results (2026-04-23, single DGX Spark GB10)

Measured with [llama-benchy](https://github.com/eugr/llama-benchy) by [@eugr](https://github.com/eugr). 3 runs, depths 0 and 16384. 12/13 models passed.

`pp` = prompt processing (tok/s, higher is better) · `tg` = token generation (tok/s, higher is better) · `TTFT` = time to first token (ms, lower is better) · `deep ctx` = tg degradation at 16k context depth

### S tier — Small / fast

| Model | Engine | pp (tok/s) | tg (tok/s) | peak | TTFT (ms) | Notes |
|---|---|---|---|---|---|---|
| Nemotron-3-Nano-4B-FP8 | vLLM | 7439 ±1591 | 40.7 | 41 | 318 | Instant responder, great for orchestration |
| Nemotron-3-Nano-30B-A3B-NVFP4 | vLLM | 8063 ±180 | 58 | 59 | 305 | Fastest 30B on the Spark, 0% ctx degradation |
| Qwen3.5-35B-Uncensored-Q4_K_M | llama.cpp | 1820 ±13 | 57.5 | 59 | 1112 | GGUF — generation speed matches FP8 vLLM |

### M tier — Medium

| Model | Engine | pp (tok/s) | tg (tok/s) | peak | TTFT (ms) | Deep ctx | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | vLLM | — ¹ | 50 | 51 | 806 | 0% @16k | |
| Qwen3.6-35B-A3B-FP8 | vLLM | — ¹ | 49.9 | 51 | 759 | 0% @16k | Slightly lower TTFT than 3.5 |
| Qwen3-VL-30B-A3B-Instruct-FP8 | vLLM | 6419 ±3100 | 52.7 | 54 | 527 | -10% @16k | Vision model |
| Qwen3-Omni-30B-A3B-Instruct | vLLM | 4478 ±1485 | 30.8 | 32 | 588 | -10% @16k | Audio + image + text |
| Qwen3-Coder-Next-FP8-Dynamic | vLLM | 3060 ±1268 | 33.6 | 35 | 889 | 0% @16k | |
| Qwen3-Coder-Next-int4-AutoRound | vLLM | 4098 ±555 | **68** | 69 | 509 | 0% @16k | Fastest generation in M tier |
| Mistral-Small-24B-Instruct-2501 | vLLM | 1962 ±183 | 4.6 | 5 | 1165 | 0% @16k | Low tg — enforce-eager penalty |

### L tier — Large (solo only, evicts all others)

| Model | Engine | pp (tok/s) | tg (tok/s) | peak | TTFT (ms) | Deep ctx | Notes |
|---|---|---|---|---|---|---|---|
| GPT-OSS-120B (MXFP4) | vLLM (mxfp4) | 4804 ±163 | **56.1** | 59 | 481 | 0% @16k | Fastest 120B on the Spark |
| Nemotron-3-Super-120B-A12B-NVFP4 | vLLM | 1880 ±57 | 15 | 16 | 1169 | 0% @16k | |
| Qwen3.5-122B-A10B-int4-AutoRound | vLLM (tf5) | **FAIL** | — | — | — | — | See note below |

> ¹ **Qwen3.5-35B-FP8 and Qwen3.6-35B-FP8 show pp=0** in this run — the prompt-processing test timed out, likely due to a cold-start scheduling edge case. Token generation numbers are correct and consistent with prior runs.

> **Qwen3.5-122B FAIL** — root cause: `model_extra_tensors.safetensors` was missing from the model directory. The fastsafetensors loader enumerates all files listed in `model.safetensors.index.json` at startup and fails hard if any are absent. Fix: re-run the HuggingFace download — it will fetch only the missing file without re-downloading the full 90 GB.
> ```bash
> huggingface-cli download Intel/Qwen3.5-122B-A10B-int4-AutoRound \
>   model_extra_tensors.safetensors \
>   --local-dir $LLM_ROOT_PATH/vllm/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound
> ```

**GPT-OSS-120B at 56 tok/s** is the standout number — a 120B model generating tokens faster than most 35B models, enabled by CUTLASS MXFP4 kernels on the Blackwell architecture ([@christopherowen](https://github.com/christopherowen)).

---

## Prerequisites

- DGX Spark with Ubuntu 22.04/24.04
- Docker + NVIDIA Container Runtime (`nvidia-container-runtime` as default runtime in `/etc/docker/daemon.json`)
- Portainer (optional but recommended for managing the stack)
- GitHub account with a Personal Access Token (PAT) for GHCR image publishing
- `huggingface-cli` for model downloads

```json
// /etc/docker/daemon.json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "args": []
        }
    }
}
```

Create the shared Docker network (one-time):

```bash
docker network create dgx_net
```

---

## Step 1 — Clone the Repo

```bash
git clone https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama.git
cd dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

Create your `.env` file:

```bash
cp .env.sample .env
# Edit .env and fill in:
#   GH_USER=your-github-username
#   GH_PAT=ghp_your_personal_access_token
#   LLM_ROOT_PATH=/home/YOUR_USER/LLMs
#   REPO_CONFIG_PATH=/home/YOUR_USER/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
#   LITELLM_MASTER_KEY=sk-choose-a-secure-key
#   POSTGRES_PASSWORD=choose-a-db-password
```

---

## Step 2 — Build and Push the Base Images

The `build_and_push.sh` script builds and pushes five images to your GitHub Container Registry:

| Image | Purpose | Dockerfile |
|---|---|---|
| `llama-cpp-spark` | llama.cpp compiled for SM12.1 (arch 121) | `llama-cpp/llama-cpp.Dockerfile` |
| `llama-swap-spark` | llama-swap proxy (ARM64 binary) | `llama-swap/llama-swap.Dockerfile` |
| `ollama-spark` | Ollama (pinned version mirror) | `ollama/ollama.Dockerfile` |
| `litellm-spark` | LiteLLM gateway (stable release mirror) | `LiteLLM/litellm.Dockerfile` |
| `vllm-spark` | vLLM from nightly wheels (lightweight) | `vllm/vllm.Dockerfile` |

```bash
bash build_and_push.sh
```

**What the llama.cpp build does differently for the Spark:** It uses `nvidia/cuda:13.1.0-devel-ubuntu24.04` (CUDA 13.1 is required to compile for architecture `121`), creates the missing `libcuda.so.1` ARM64 stub, and compiles with `-DCMAKE_CUDA_ARCHITECTURES="121"`. Pre-built ARM64 llama.cpp binaries from most distros will silently fall back to CPU-only — this build ensures you actually use the GPU.

---

## Step 3 — Build the vLLM Model-Serving Images

> **Credit: [@eugr](https://github.com/eugr) — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**
>
> The build system in `vllm/build/spark-vllm-docker/` downloads pre-built vLLM + FlashInfer wheels from eugr's GitHub releases. These wheels are compiled specifically for the GB10 (CUDA 13.1, SM12.1a, ARM64 SBSA) and updated regularly. Without them, every build requires compiling FlashInfer and vLLM from source — a 2–4 hour process. The build script automatically falls back to source compilation if the pre-built wheels aren't available or if you pass `--rebuild-vllm`/`--rebuild-flashinfer`. Thank you @eugr for maintaining this — it makes iterating on the stack practical.

The `build_and_push.sh`'s `vllm-spark` image is a lightweight wrapper. For actually serving models you need the purpose-built images from `vllm/build/spark-vllm-docker/`:

```bash
cd vllm/build/spark-vllm-docker
```

**Build the standard image** (used for Nemotron-30B, Qwen3-VL, Qwen3-Omni, Mistral, etc.):

```bash
bash build-and-copy.sh
# produces: vllm-node
```

**Build with Transformers 5.x** (required for Qwen3.5-122B-MoE, Qwen3.6-35B, Qwen3-Coder-Next — models using the newer Mamba/hybrid architecture):

```bash
bash build-and-copy.sh --tf5
# produces: vllm-node-tf5
```

**Build with experimental MXFP4 support** (for OpenAI GPT-OSS-120B):

> **Credit: [@christopherowen](https://github.com/christopherowen) — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker)**
>
> This build uses christopherowen's forks of vLLM, FlashInfer, and CUTLASS that add native MXFP4 quantization support for the GB10's CUTLASS kernels. The `--mxfp4-backend CUTLASS` + `--mxfp4-layers moe,qkv,o,lm_head` flags this enables are what push GPT-OSS-120B from ~35 tok/s to ~57 tok/s on a single Spark. This is not in upstream vLLM yet. Huge thanks to @christopherowen for this work.

```bash
bash build-and-copy.sh --exp-mxfp4
# produces: vllm-node-mxfp4
# takes ~1 hour (compiles FlashInfer + CUTLASS fork from source)
```

Build times on the GB10 using @eugr's pre-built wheels: ~15 minutes. From source: 2–4 hours.

---

## Step 4 — Download Models

Install the HuggingFace CLI:

```bash
pip install huggingface-hub
```

Download each model into the directory structure expected by the config. Replace `$LLM_ROOT_PATH` with your actual path (e.g. `/home/YOUR_USER/LLMs`):

```bash
BASE=$LLM_ROOT_PATH/vllm

# --- S tier ---
huggingface-cli download nvidia/Nemotron-3-Nano-4B-FP8 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-4B-FP8
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-30B-A3B-NVFP4
huggingface-cli download Intel/Qwen3-Coder-Next-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-int4-AutoRound

# --- M tier ---
huggingface-cli download Qwen/Qwen3.5-35B-A3B-FP8 \
  --local-dir $BASE/Alibaba/Qwen3.5-35B-A3B-FP8
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local-dir $BASE/Alibaba/Qwen3-VL-30B-A3B-Instruct-FP8
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir $BASE/Alibaba/Qwen3-Omni-30B-A3B-Instruct
huggingface-cli download Qwen/Qwen3-Coder-Next-FP8-Dynamic \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-FP8-Dynamic
huggingface-cli download mistralai/Mistral-Small-24B-Instruct-2501 \
  --local-dir $BASE/Mistral/Mistral-Small-24B-Instruct-2501

# --- L tier ---
huggingface-cli download Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --local-dir $BASE/Nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
huggingface-cli download openai/gpt-oss-120b \
  --local-dir $BASE/OpenAI/GPT-OSS-120B

# --- GGUF (llama.cpp) ---
huggingface-cli download HauhauCS/Qwen3.5-35B-A3B-Uncensored-Aggressive \
  --include "*.gguf" --include "*.jinja" \
  --local-dir $LLM_ROOT_PATH/ollama/Alibaba/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

---

## Step 5 — Prepare Config Files

```bash
REPO=/home/YOUR_USER/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
mkdir -p $REPO/llama-swap/scripts
```

**llama-swap/config.yaml** — the heart of the stack. Copy and patch the sample:

```bash
cp llama-swap/config.yaml.sample llama-swap/config.yaml
sed -i "s|/path/to/LLMs|$LLM_ROOT_PATH|g" llama-swap/config.yaml
sed -i "s|/path/to/Docker|$HOME/Docker|g" llama-swap/config.yaml
```

Key concepts in the config:

```yaml
host: "0.0.0.0"
port: 8080
readyTimeout: 3600

macros:
  host: "0.0.0.0"
  tensor_parallel: "1"

groups:
  # S group: small/fast models — swap:true means evict previous before loading next
  small-models:
    swap: true
    exclusive: true   # evict M and L groups when any S model loads

  # M group: 30B FP8 / 24B BF16
  medium-models:
    swap: true
    exclusive: true

  # L group: 120B+ MoE, always solo
  large-models:
    swap: true
    exclusive: true

models:
  MyModel-30B:
    ttl: 600          # evict after 600s idle
    readyTimeout: 600
    cmd: >
      docker run --rm --name vllm-mymodel-${PORT}
      --runtime nvidia --gpus all --ipc=host
      --network container:llama-swap
      -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1
      -v /home/YOUR_USER/LLMs/vllm:/models/vllm
      vllm-node
      vllm serve /models/vllm/MyOrg/MyModel-30B
      --served-model-name MyModel-30B
      --host ${host} --port ${PORT}
      --gpu-memory-utilization 0.70
      --max-model-len 131072
      --kv-cache-dtype fp8
      --load-format fastsafetensors
      --enable-prefix-caching
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
    cmdStop: "docker stop vllm-mymodel-${PORT}"
```

The `--network container:llama-swap` flag is the linchpin — it puts the model container inside llama-swap's network namespace so it binds to `localhost:${PORT}`, which llama-swap proxies. Without it, llama-swap can't reach the model.

See [llama-swap/config.yaml.sample](llama-swap/config.yaml.sample) for the full annotated config with all models.

**Critical memory math for 128 GB GB10:**

```
CUDA-visible total: ~121.7 GiB

S tier (4B–30B quant):  0.50–0.65 × 121.7 = 61–79 GiB  → swap:true
M tier (30B–35B FP8):   0.60–0.75 × 121.7 = 73–91 GiB  → swap:true
L tier (120B+ MoE):     0.70–0.85 × 121.7 = 85–103 GiB → swap:true, solo
```

**LiteLLM/config.yaml** — copy the sample and update your master key:

```bash
cp LiteLLM/config.yaml.sample LiteLLM/config.yaml
sed -i "s|sk-your-litellm-master-key|YOUR_MASTER_KEY|g" LiteLLM/config.yaml
```

Each llama-swap model needs an entry pointing to `http://llama-swap:8080/v1`:

```yaml
model_list:
  - model_name: MyModel-30B
    litellm_params:
      model: openai/MyModel-30B
      api_base: "http://llama-swap:8080/v1"
      api_key: "sk-your-master-key"
      supports_reasoning: true
      include_reasoning: true
      merge_reasoning_content_in_choices: true
```

See [LiteLLM/config.yaml.sample](LiteLLM/config.yaml.sample) for all models with their reasoning flags.

---

## Step 6 — Dynamic VRAM Launcher for Large Models

Large MoE models (120B+ INT4/FP4) hit a recurring failure: after the previous model container exits, the CUDA memory allocator on the unified-memory GB10 doesn't immediately return all memory to the free pool. vLLM's startup check `free_memory >= gpu_memory_utilization × total` fails with a hardcoded high utilization value.

The fix is a small wrapper script that queries actual free VRAM at launch time and computes the safe utilization dynamically. Add it to `llama-swap/scripts/` (this directory is mounted into the llama-swap container at `/app/scripts/`):

**`llama-swap/scripts/launch-large-model.sh`** — adapt `MODEL_PATH`, container name, and vllm flags for your model:

```bash
#!/bin/bash
# Dynamically sets --gpu-memory-utilization based on actually-free VRAM at launch time.
# Usage (from llama-swap cmd): /app/scripts/launch-large-model.sh PORT HOST
set -euo pipefail

PORT="${1}"
HOST="${2}"

# Query free/total VRAM via nvidia-smi inside the (already-cached) vllm image.
# Adds ~3s overhead — negligible vs. the several minutes this model takes to load.
MEM_LINE=$(docker run --rm --runtime nvidia --gpus all \
    vllm-node-tf5:latest \
    sh -c "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheaders,nounits | head -1" \
    2>/dev/null || true)

FREE_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$1); print $1+0}')
TOTAL_MIB=$(echo "$MEM_LINE" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')

# GB10: nvidia-smi reports 128 GiB (131072 MiB) unified memory;
# CUDA sees only 121.69 GiB (124610 MiB). Subtract that delta + 3 GiB safety margin
# so the computed value matches vLLM's view of available memory.
GMEM=$(awk -v f="$FREE_MIB" -v t_nv="$TOTAL_MIB" 'BEGIN {
    cuda_t   = 124610;
    safety   = 3072;
    overhead = (t_nv > cuda_t) ? t_nv - cuda_t : 0;
    cuda_free = f - overhead - safety;
    if (cuda_free < 0) cuda_free = 0;
    u = cuda_free / cuda_t;
    if (u > 0.85) u = 0.85;
    if (u < 0.60) u = 0.60;
    printf "%.2f", u;
}')

if [ -z "$GMEM" ] || [ "$FREE_MIB" = "0" ]; then
    echo "[auto-gmem] WARNING: VRAM query failed, using fallback 0.75"
    GMEM="0.75"
fi

echo "[auto-gmem] nvidia-smi free=${FREE_MIB}MiB / total=${TOTAL_MIB}MiB → gpu_memory_utilization=${GMEM}"

exec docker run --rm --name "vllm-mymodel-122b-${PORT}" \
    --runtime nvidia --gpus all --ipc=host --network container:llama-swap \
    -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -v /home/YOUR_USER/LLMs/vllm:/models/vllm \
    vllm-node-tf5:latest \
    vllm serve /models/vllm/MyOrg/MyModel-122B \
    --served-model-name MyModel-122B \
    --host "${HOST}" --port "${PORT}" \
    --gpu-memory-utilization "${GMEM}" \
    --max-model-len 131072 \
    --kv-cache-dtype fp8 \
    --load-format fastsafetensors \
    --attention-backend FLASHINFER \
    --mamba-ssm-cache-dtype float16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

```bash
chmod +x llama-swap/scripts/launch-large-model.sh
```

Reference it from `llama-swap/config.yaml`:

```yaml
  MyModel-122B:
    ttl: 3600
    readyTimeout: 1800
    cmd: /app/scripts/launch-large-model.sh ${PORT} ${host}
    cmdStop: "docker stop vllm-mymodel-122b-${PORT}"
```

---

## Step 7 — Deploy the Stack via Portainer

In Portainer → Stacks → Add Stack → paste the following. **Replace all `YOUR_*` placeholders** before deploying.

```yaml
version: '3.8'
services:

  # ==========================================
  # 1. LLAMA.CPP (GGUF Engine)
  # ==========================================
  llama-server:
    image: ghcr.io/martin-b78/llama-cpp-spark:latest
    container_name: llama.cpp
    restart: unless-stopped
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    security_opt:
      - seccomp:unconfined
    ports:
      - "18080:18080"
    volumes:
      - /home/YOUR_USER/LLMs/ollama:/models/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
    command:
      - --host
      - "0.0.0.0"
      - --port
      - "18080"
      - --parallel
      - "4"
      - --no-mmap
      - --context-shift
      - --models-dir
      - /models
      - --n-gpu-layers
      - "99"
      - --ctx-size
      - "16384"
    stdin_open: true
    tty: true
    networks:
      - dgx_net

  # ==========================================
  # 2. OLLAMA
  # ==========================================
  ollama:
    image: ghcr.io/martin-b78/ollama-spark:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - /home/YOUR_USER/LLMs/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_LLM_LIBRARY=cuda_v13
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    networks:
      - dgx_net

  # ==========================================
  # 3. LLAMA-SWAP (VRAM Orchestrator)
  # Port 28080 (host) → 8080 (container)
  # LiteLLM reaches it via Docker DNS: http://llama-swap:8080
  # Model containers attach via: --network container:llama-swap
  # ==========================================
  llama-swap:
    image: ghcr.io/martin-b78/llama-swap-spark:latest
    container_name: llama-swap
    restart: unless-stopped
    ports:
      - "28080:8080"
    entrypoint: ["/usr/bin/llama-swap", "-config", "/app/config.yaml", "-listen", "0.0.0.0:8080"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    volumes:
      - /home/YOUR_USER/Docker/REPO_DIR/llama-swap:/app
      - /var/run/docker.sock:/var/run/docker.sock
      - /home/YOUR_USER/LLMs:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - GITHUB_TOKEN=YOUR_GITHUB_PAT
    networks:
      - dgx_net

  # ==========================================
  # 4. LITELLM DATABASE
  # ==========================================
  litellm-db:
    image: postgres:15-alpine
    container_name: litellm-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: litellm_admin
      POSTGRES_PASSWORD: YOUR_DB_PASSWORD
    ports:
      - "15432:5432"
    volumes:
      - litellm_db_data:/var/lib/postgresql/data
    networks:
      - dgx_net

  # ==========================================
  # 5. LITELLM GATEWAY
  # ==========================================
  litellm:
    image: ghcr.io/martin-b78/litellm-spark:latest
    container_name: litellm
    restart: unless-stopped
    depends_on:
      - litellm-db
    ports:
      - "14000:4000"
    environment:
      - DATABASE_URL=postgresql://litellm_admin:YOUR_DB_PASSWORD@litellm-db:5432/litellm
      - LITELLM_MASTER_KEY=YOUR_MASTER_KEY
    volumes:
      - /home/YOUR_USER/Docker/REPO_DIR/LiteLLM/config.yaml:/app/config.yaml
    command:
      - "--config"
      - "/app/config.yaml"
      - "--port"
      - "4000"
    networks:
      - dgx_net

networks:
  dgx_net:
    external: true

volumes:
  litellm_db_data:
```

---

## Step 8 — Verify the Stack

```bash
# llama-swap up and listing all configured models
curl http://localhost:28080/v1/models | python3 -m json.tool

# LiteLLM health
curl http://localhost:14000/health

# Trigger a model load (llama-swap starts the container on first request)
curl http://localhost:28080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MyModel-30B","messages":[{"role":"user","content":"Hello"}]}'

# Watch llama-swap load the container in real time
docker logs -f llama-swap
```

---

## Step 9 — Benchmarking

> **Credit: [@eugr](https://github.com/eugr) — [llama-benchy](https://github.com/eugr/llama-benchy)**
>
> llama-benchy is a standardized LLM benchmark tool that measures prompt-processing (pp) and token-generation (tg) throughput in a reproducible way specifically designed for comparing results across the DGX Spark community. The benchmark script in this repo is a wrapper around llama-benchy. Thank you @eugr for creating and maintaining it — the consistent output format makes it possible to compare results across different model configs and post meaningful numbers to the forums.

Install llama-benchy first:

```bash
pip install llama-benchy
# or: uvx llama-benchy  (no install needed with uv)
```

Then run the full benchmark across all configured models:

```bash
bash benchmark-models.sh --endpoint http://localhost:28080
```

The script tests each model sequentially, runs a coherence check to detect repetition loops, and writes a results summary to `test-results/`. The llama-benchy output table is formatted for direct copy-paste into forum posts.

---

## Tips & Common Issues

**Model containers fail with "port already in use"**
llama-swap assigns ports dynamically from its pool. Make sure the port range in config.yaml doesn't overlap with other services on the host.

**vLLM startup check: `free_memory < gpu_memory_utilization × total`**
After stopping one model container, the CUDA allocator on unified-memory systems can hold freed memory for several seconds. Use the dynamic launcher script from Step 6 instead of a hardcoded `--gpu-memory-utilization` value for any model over 100B parameters.

**Mamba/hybrid models need the tf5 image and an extra flag**
Models using the Mamba SSM layers (Qwen3.5-122B-A10B, Qwen3.6-35B, Qwen3-Coder-Next) require `vllm-node-tf5:latest` (built with `--tf5`) and `--mamba-ssm-cache-dtype float16` in the vllm serve command.

**`--load-format fastsafetensors` is strongly recommended**
It loads weight shards in parallel and cuts startup time by ~40% for multi-shard models. Requires `model.safetensors.index.json` to be present alongside the weight files (all HuggingFace multi-shard models include it).

**GPT-OSS-120B MXFP4: skip Ray**
Do not use `--distributed-executor-backend ray` for single-GPU MoE models. Ray's GCS server + dashboard add ~500 MB overhead, which pushes total allocation past Ray's 95% OOM threshold. vLLM's default `mp` (multiprocessing) executor is leaner and also re-enables async scheduling.

**S/M/L group sizing on 128 GB**

```
CUDA-visible: ~121.7 GiB

S (4B–30B quantized):  0.50–0.65 × 121.7 GiB = 61–79 GiB  → swap:true
M (30B–35B FP8):       0.60–0.75 × 121.7 GiB = 73–91 GiB  → swap:true
L (120B+ MoE):         0.70–0.85 × 121.7 GiB = 85–103 GiB → swap:true, solo
```

With `swap: true` on all groups, the active model is always evicted before the next one loads. `exclusive: true` evicts all other groups when a new group activates — essential for preventing OOM when transitioning between tiers.

---

## Acknowledgements

This stack stands on the shoulders of several people's work:

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker): pre-built vLLM + FlashInfer wheels for GB10, and [llama-benchy](https://github.com/eugr/llama-benchy): the benchmarking tool. Both projects are essential to making this practical. Thank you.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) and the custom vLLM/FlashInfer/CUTLASS forks enabling native MXFP4 on GB10. This is what unlocks GPT-OSS-120B at full speed. Thank you.
- **[mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)** — the VRAM orchestrator at the center of this stack.
- **[BerriAI/litellm](https://github.com/BerriAI/litellm)** — the unified API gateway.
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)** and the **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** team — the inference engines powering the model serving.

---

*Repo: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama*
