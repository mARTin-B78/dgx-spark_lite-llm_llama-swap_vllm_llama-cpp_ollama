# Running a Full Multi-Model LLM Stack on DGX Spark (GB10) — With VRAM Orchestration

> ⚠️ **Two Setup Approaches Available:**
> 
> **New users:** Use the automated setup wizard at `./setup/setup.sh` (~5 minutes) — it handles Docker detection, service configuration, credential collection, and automatic file generation.
> 
> **Advanced users / Full control:** Follow the detailed manual steps in this guide (~45 minutes) with complete explanations and customization options.
> 
> Both approaches produce the same working stack. Choose based on your experience level and preference.

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
- `huggingface-hub` for model downloads (via `hf download` command-line tool, or `pip install huggingface-hub[cli]`)

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

## Two Setup Paths

### 🚀 Quick Start: Automated Setup (~5 minutes)

If you want to get started quickly without worrying about configuration details, use the interactive setup wizard:

```bash
git clone https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama.git
cd dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

# Run the setup wizard
./setup/setup.sh

# Download models automatically
./setup/download-models.sh

# Start the stack
docker compose up -d
```

The setup wizard will:
- Detect your Docker installation and NVIDIA runtime
- Check running services and available ports
- Offer to resolve any port conflicts
- Collect your credentials (HuggingFace token, GitHub PAT)
- Let you select which model tiers to download
- Auto-generate `.env` and `docker-compose.yml`
- Create a configuration summary

**This is the recommended path for first-time users.** For documentation and troubleshooting, see [setup/README.md](setup/README.md).

---

### 📖 Detailed Walkthrough: Manual Setup (This Guide)

If you want to understand each component deeply, learn how to customize configurations, or prefer step-by-step control, follow the instructions below. This guide walks through:

1. 9 detailed setup steps with explanations
2. Model tier system and architecture details
3. Troubleshooting and tips

**This is the recommended path for advanced users and those who want complete control over configuration.**

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

### About GitHub Personal Access Token (PAT)

You need a GitHub PAT to push Docker images to GitHub Container Registry (GHCR). **Why?**
- `build_and_push.sh` builds 5 base Docker images and publishes them to your GHCR account
- This lets you store pre-built images so you don't have to recompile from scratch
- Other users/machines can then pull your images

**What permissions does it need?**
- Scope: `write:packages` (to push container images)
- No need for repo code access — only container registry write permissions

**Can you skip it?**
Yes, if you want to build everything locally and not use GHCR. Edit `build_and_push.sh` and replace the push step with a local tag (e.g., `docker tag llama-cpp-spark llama-cpp-spark:latest`). However, this means each model launcher will need a clone of this repo.

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

The `build_and_push.sh`'s `vllm-spark` image is a lightweight wrapper. For actually serving models you need the purpose-built images from `vllm/build/spark-vllm-docker/`.

**Note:** If the `vllm/build/spark-vllm-docker/` folder doesn't exist after cloning, you need to fetch eugr's spark-vllm-docker submodule:

```bash
# From the repo root, initialize submodules
git submodule update --init --recursive

# Then navigate to the build folder
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
pip install huggingface-hub[cli]
```

Download each model into the directory structure expected by the config. Replace `$LLM_ROOT_PATH` with your actual path (e.g. `/home/YOUR_USER/LLMs`).

**Note:** `huggingface-cli` is deprecated. Use `hf download` instead:

```bash
BASE=$LLM_ROOT_PATH/vllm

# --- S tier ---
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-4B-FP8
hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-30B-A3B-NVFP4
hf download Intel/Qwen3-Coder-Next-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-int4-AutoRound

# --- M tier ---
hf download Qwen/Qwen3.5-35B-A3B-FP8 \
  --local-dir $BASE/Alibaba/Qwen3.5-35B-A3B-FP8
hf download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local-dir $BASE/Alibaba/Qwen3-VL-30B-A3B-Instruct-FP8
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir $BASE/Alibaba/Qwen3-Omni-30B-A3B-Instruct
hf download unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-FP8-Dynamic
hf download mistralai/Mistral-Small-24B-Instruct-2501 \
  --local-dir $BASE/Mistral/Mistral-Small-24B-Instruct-2501

# --- L tier ---
hf download Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --local-dir $BASE/Nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
hf download openai/gpt-oss-120b \
  --local-dir $BASE/OpenAI/GPT-OSS-120B

# --- GGUF (llama.cpp) ---
# Downloads all quantized variants (100+ GB). For faster setup, pick ONE quant variant:
hf download HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive \
  --include "*Q4_K_M*" --include "*.jinja" \
  --local-dir $LLM_ROOT_PATH/ollama/Alibaba/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

**Tip for GGUF models:** The `--include` filter prevents downloading all quantization variants (Q2, Q3, Q4, Q5, Q6, etc.). Using `*Q4_K_M*` downloads only the Q4_K_M (medium) variant, which balances quality and VRAM.

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

## Step 6 — Dynamic VRAM Launcher (any vLLM model)

Three failure modes motivate a wrapper:

1. **Residual CUDA memory after model swap.** On unified-memory GB10, after the previous container exits the CUDA allocator can hold tens of GiB for several seconds. vLLM's startup check `free_memory >= gpu_memory_utilization × total` then fails with a hardcoded utilization.
2. **Wrong static value across reboots / different co-resident loads.** A value tuned for an empty GPU is too high on a warm one (and vice versa).
3. **System-RAM pressure.** GB10 unified memory becomes unstable above ~126.5 GB used. A vLLM pool that fits in CUDA's view can still push total system usage past that crash threshold when other workloads consume RAM.

The repo ships a **generic adaptive launcher** at [llama-swap/scripts/launch-vllm-auto.sh](llama-swap/scripts/launch-vllm-auto.sh) that:

1. Estimates **weights** from the sum of `*.safetensors` in the model dir.
2. Estimates **KV cache** from `config.json` (handles nested `text_config` for multimodal models):
   `kv_bytes ≈ 2 × num_hidden_layers × num_kv_heads × head_dim × max_model_len × max_num_seqs × kv_dtype_bytes`
3. Adds a `SAFETY_GIB` headroom → `need_gib`.
4. Reads `MemTotal` and `MemAvailable` from `/proc/meminfo` (works on GB10 where `nvidia-smi --query-gpu=memory.*` returns "Not Supported").
5. Caps at `SYSTEM_RAM_CEILING_GIB` (default 117.81 GiB = 126.5 GB decimal) so `(MemTotal − ceiling)` GiB stays reserved no matter what the kernel reports as available.
6. Picks `util = need / total`, clamps to `[GMEM_MIN, GMEM_MAX]`, additionally caps at `(free − GMEM_FREE_BUFFER_GIB) / total` (default 5 GiB buffer to bridge the `MemAvailable` vs `cudaMemGetInfo` race at vLLM startup). Aborts if not even the cap fits.

**Adaptive vs static — single env var:** set `GMEM_OVERRIDE` to a number (e.g. `0.7069`) to pin gpu_memory_utilization to that exact value and skip the calculation. Set it to `adaptive` (or leave it unset) to compute dynamically. One env-var flip switches a model between hand-tuned and adaptive without restructuring the block — useful when one specific model needs thermal-conscious pinning.

This means every vLLM block can use the same launcher template; the only difference is which envs you set.

**Wire it from `llama-swap/config.yaml`:**

```yaml
  Qwen3.6-35B-A3B-FP8:
    ttl: 600
    readyTimeout: 600
    checkEndpoint: "/health"
    cmd: >
      env
      MODEL_PATH=/models/vllm/Alibaba/Qwen3.6-35B-A3B-FP8
      MODEL_HOST_PATH=/home/sparky/LLMs/vllm/Alibaba/Qwen3.6-35B-A3B-FP8
      CONTAINER_NAME=vllm-qwen3.6-35b-${PORT}
      IMAGE=vllm-node-tf5:latest
      PORT=${PORT} HOST=${host}
      MAX_MODEL_LEN=131072 MAX_NUM_SEQS=10 KV_DTYPE_BYTES=1
      GMEM_MIN=0.55 GMEM_MAX=0.85 SAFETY_GIB=4
      /app/scripts/launch-vllm-auto.sh
      --served-model-name Qwen3.6-35B-A3B-FP8
      --chat-template /models/vllm/Alibaba/Qwen3.6-35B-A3B-FP8/chat_template-tool-strict.jinja
      --max-num-batched-tokens 32768
      --max-cudagraph-capture-size 10
      --kv-cache-dtype fp8
      --load-format fastsafetensors
      --attention-backend FLASHINFER
      --enable-prefix-caching
      --trust-remote-code
      --mamba-ssm-cache-dtype float16
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --reasoning-parser qwen3
      --default-chat-template-kwargs '{"enable_thinking": true}'
    cmdStop: "docker stop vllm-qwen3.6-35b-${PORT}"
```

**Required env vars:** `MODEL_PATH` (in-container), `MODEL_HOST_PATH` (host — used to read `config.json` and stat safetensors), `CONTAINER_NAME`, `IMAGE`, `PORT`, `HOST`.

**Adaptive bounds:** `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `KV_DTYPE_BYTES` (1 = fp8, 2 = bf16/fp16), `GMEM_MIN`, `GMEM_MAX`, `SAFETY_GIB`.

**System guards:** `SYSTEM_RAM_CEILING_GIB` (default 117.81 = 126.5 GB), `GMEM_FREE_BUFFER_GIB` (default 5), `CUDA_OVERHEAD_GIB` (default 6.3 for GB10).

**Image plumbing:** `EXTRA_DOCKER_ARGS` (extra mounts/envs as a single space-separated string), `PRE_LAUNCH_CMD` (in-container patch step run before `vllm serve`), `VLLM_SERVE_PREFIX` (set to `""` for `vllm/vllm-openai`-style images whose ENTRYPOINT is already `vllm serve`).

**Static-pin mode:** `GMEM_OVERRIDE=0.7069` (or any 0–1 number) pins gpu_memory_utilization and skips the calculation. Unset / empty / `adaptive` = compute dynamically.

Any args after the script name are forwarded verbatim to `vllm serve`.

**Worked example (Qwen3.6-35B-A3B-FP8):** weights = 29.4 GiB; layers = 40, kv_heads = 2, head_dim = 256, ctx = 131072, batch = 10, fp8 KV → kv ≈ 50 GiB; need = 29.4 + 50 + 4 = 83.4 GiB; on 121.69 GiB total → `util ≈ 0.69` (vs. the brittle hardcoded 0.78 that tripped startup when only 91.76 GiB was free).

**122B-specific launcher.** [llama-swap/scripts/launch-qwen35-122b.sh](llama-swap/scripts/launch-qwen35-122b.sh) is the older, model-specific variant kept in place for the INT4 122B model. The generic launcher above can replace it; the dedicated one is left as a known-good reference.

```yaml
  Qwen3.5-122B-A10B-int4-AutoRound:
    ttl: 3600
    readyTimeout: 1800
    cmd: /app/scripts/launch-qwen35-122b.sh ${PORT} ${host}
    cmdStop: "docker stop vllm-qwen3.5-122b-${PORT}"
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

> **Credits**
> - **[@eugr](https://github.com/eugr) — [llama-benchy](https://github.com/eugr/llama-benchy)**: standardized throughput benchmark (pp/tg/TTFT) shared across the DGX Spark community. The output table is formatted for direct copy-paste into forum posts.
> - **[@SeraphimSerapis](https://github.com/SeraphimSerapis) — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)**: 69-scenario tool-calling quality benchmark (tool selection, parameter precision, multi-step chains, safety/refusal, structured output). Optional, opt-in via `--quality`. See the [forum announcement](https://forums.developer.nvidia.com/t/introducing-tool-eval-bench-cli/366903).
>
> `benchmark-models.sh` wraps both. When `--quality` is on, llama-benchy and tool-eval-bench run back-to-back against the *same* loaded model so the costly load step happens only once per model.

### Speed only (default)

```bash
pip install llama-benchy
# or: uvx llama-benchy  (no install needed with uv)

bash benchmark-models.sh                       # default = "Medium Log" profile
bash benchmark-models.sh --quick Qwen3.6       # smoke test, single model
bash benchmark-models.sh --stress              # adds depth 32768
bash benchmark-models.sh --extreme             # adds depth 65535
bash benchmark-models.sh --arena               # spark-arena leaderboard profile
```

### Speed + tool-calling quality

Per model the loop becomes: **load → coherence check → llama-benchy (pp/tg/depth sweep) → tool-eval-bench (tool-call scenarios) → unload**. The expensive load happens once.

```bash
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git

bash benchmark-models.sh --quality                                  # 15-scenario short pass
bash benchmark-models.sh --quality --quality-mode full              # full 69 scenarios
bash benchmark-models.sh --quality --quality-mode hardmode          # full + 5 adversarial
bash benchmark-models.sh --quality --quality-categories "K A J"     # selected categories only
bash benchmark-models.sh --quick --quality Qwen3.6-35B-A3B-FP8      # combine with any speed profile
```

When `--quality` is enabled the summary table grows a `Quality /100` column and per-model markdown reports land in `test-results/quality/<run_id>/report.md`. The script tests each model sequentially, runs a coherence check to detect repetition loops, and writes a combined results summary to `test-results/benchmarks/`.

> **Note on `groups:`**: if you previously enabled the `groups:` block in `llama-swap/config.yaml` and see the wrong model loading mid-benchmark, comment out the entire `groups:` block. Group eviction logic can swap to a sibling model when one fails to load, which corrupts the run (the wrong model gets benchmarked). See the sample config for the disabled-by-default layout.

---

## Tips & Common Issues

**Model containers fail with "port already in use"**
llama-swap assigns ports dynamically from its pool. Make sure the port range in config.yaml doesn't overlap with other services on the host.

**vLLM startup check: `free_memory < gpu_memory_utilization × total`**
Switch the model block to [llama-swap/scripts/launch-vllm-auto.sh](llama-swap/scripts/launch-vllm-auto.sh) (Step 6) — it sizes utilization from the model's actual weights+KV need vs. currently free RAM and clamps to `[GMEM_MIN, GMEM_MAX]`. The default `GMEM_FREE_BUFFER_GIB=5` covers the small race between `MemAvailable` and `cudaMemGetInfo` at vLLM startup. Want a hand-tuned static value instead? Set `GMEM_OVERRIDE=0.7069` (or whichever number) — same launcher, single env-var flip.

**System crashes near 126.5 GB used RAM**
GB10 unified memory becomes unstable above this point. The launcher defaults `SYSTEM_RAM_CEILING_GIB=117.81` (= 126.5 GB decimal) so its calculation always reserves `(MemTotal − ceiling)` GiB even when `/proc/meminfo` says more is technically available. Override per-block with `SYSTEM_RAM_CEILING_GIB=...` if a specific workload needs more or less headroom.

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
- **[@SeraphimSerapis](https://github.com/SeraphimSerapis)** — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): tool-calling quality benchmark with 69 scenarios across selection, parameter precision, multi-step chains, safety, and structured output. Wired into `benchmark-models.sh` via `--quality`. Thank you.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) and the custom vLLM/FlashInfer/CUTLASS forks enabling native MXFP4 on GB10. This is what unlocks GPT-OSS-120B at full speed. Thank you.
- **[mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)** — the VRAM orchestrator at the center of this stack.
- **[BerriAI/litellm](https://github.com/BerriAI/litellm)** — the unified API gateway.
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)** and the **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** team — the inference engines powering the model serving.

---

*Repo: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama*
