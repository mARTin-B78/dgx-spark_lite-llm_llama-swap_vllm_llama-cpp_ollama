# DGX Spark AI Orchestration Stack

> **Full-stack LLM inference for NVIDIA DGX Spark (Grace-Blackwell GB10)**
> vLLM · llama.cpp · llama-swap · Ollama · LiteLLM — on 128 GB unified memory

---

## What is this?

A production-ready Docker Compose stack that lets a single DGX Spark run **multiple large language models** without manual VRAM juggling. `llama-swap` acts as the orchestrator: it spins up the right inference container on demand and evicts it when idle, so 128 GB is never wasted on a model nobody is using.

A unified **LiteLLM** gateway exposes every model through one OpenAI-compatible endpoint with a single API key — no per-service port juggling.

---

## Stack overview

```
Client (Claude Code / Open WebUI / curl)
        │
        ▼  :14000 (OpenAI-compatible)
  ┌─────────────┐
  │   LiteLLM   │  ◄── unified gateway, auth, routing
  └──────┬──────┘
         │ routes by model name
         ▼ :28080
  ┌─────────────┐
  │ llama-swap  │  ◄── VRAM orchestrator (docker.sock)
  └──────┬──────┘
         │ spawns on demand
    ┌────┼──────────────────────┐
    ▼    ▼                      ▼
 vllm  llama.cpp             Ollama
 :PORT :PORT                 :11434
(ephemeral model containers)
```

| Service | Host port | Role |
|---|---|---|
| LiteLLM | 14000 | API gateway |
| llama-swap | 28080 | VRAM orchestrator |
| Ollama | 11434 | GGUF / Ollama models |
| llama.cpp (persistent) | 19000 | GGUF engine |
| vLLM (persistent) | 18000 | Safetensors engine |
| LiteLLM DB (Postgres) | 15432 | LiteLLM backend |

---

## Model tier system

llama-swap groups models into tiers so concurrent loading is safe on 128 GB unified memory (~108 GB available after OS):

| Tier | gpu_mem | VRAM each | Max concurrent | Examples |
|---|---|---|---|---|
| **S** Small | 0.12–0.22 | 15–28 GB | 4× | Nemotron-4B-FP8, Qwen3.5-35B-GGUF |
| **M** Medium | 0.40 | ~51 GB | 2× | Qwen3.5-35B-FP8, Mistral-Small-24B |
| **L** Large | 0.70–0.85 | ~90–109 GB | 1× (solo) | Qwen3.5-122B, Nemotron-120B, GPT-OSS-120B |

`exclusive: true` on the L-tier means loading any large model automatically evicts all others.

---

## Prerequisites

- NVIDIA DGX Spark (GB10) running Ubuntu 24.04
- Docker with NVIDIA Container Toolkit (`nvidia-ctk`)
- A GitHub Container Registry account (GHCR) to host your images
- GitHub Personal Access Token (PAT) with `write:packages` scope

---

## Step 1 — Create the Docker network

All containers share one bridge network so they can resolve each other by name (e.g. `http://llama-swap:8080`):

```bash
docker network create dgx_net
```

---

## Step 2 — Create `.env`

Copy the sample and fill in your values:

```bash
cp .env.sample .env
```

```dotenv
# .env
GH_USER=your-github-username
IMAGE_TAG=latest

LLM_ROOT_PATH=/home/YOUR_USER/LLMs
REPO_CONFIG_PATH=/home/YOUR_USER/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

LITELLM_MASTER_KEY=sk-choose-a-secure-key
POSTGRES_PASSWORD=choose-a-db-password
DATABASE_URL="postgresql://litellm_admin:choose-a-db-password@127.0.0.1:15432/litellm"
```

---

## Step 3 — Build `vllm-node` (standard vLLM)

Used for: Qwen3.5-35B-FP8, Mistral-Small-24B, Nemotron-4B-FP8, Nemotron-30B-NVFP4, GPT-OSS-120B, Qwen3-VL, Qwen3-Omni.

```bash
cd vllm/build/spark-vllm-docker
./build-and-copy.sh
# Image tag defaults to: vllm-node
```

---

## Step 4 — Build `vllm-node-tf5` (transformers v5, Mamba/hybrid models)

Required for: Qwen3.5-122B-A10B-int4-AutoRound, Qwen3-Coder-Next, and any other hybrid Mamba architecture.

```bash
./build-and-copy.sh --tf5
# Image tag defaults to: vllm-node-tf5
```

---

## Step 5 — Build `vllm-node-mxfp4` (CUTLASS MXFP4 kernels)

Optional. Only needed for GPT-OSS-120B with native MXFP4 quantization (fastest path, ~60 tok/s).

```bash
./build-and-copy.sh --exp-mxfp4
# Image tag defaults to: vllm-node-mxfp4
```

---

## Step 6 — Download models

Install the HuggingFace CLI if you don't have it:

```bash
pip install huggingface-hub
```

Then download each model into the directory structure that matches `LLM_ROOT_PATH`. The paths below correspond to the `config.yaml.sample` — change the base prefix to your `$LLM_ROOT_PATH`.

### vLLM models (safetensors)

```bash
BASE=$LLM_ROOT_PATH/vllm   # e.g. /home/YOUR_USER/LLMs/vllm

# --- S tier (small / fast) ---
huggingface-cli download nvidia/Nemotron-3-Nano-4B-FP8 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-4B-FP8

huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-30B-A3B-NVFP4

huggingface-cli download Intel/Qwen3-Coder-Next-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-int4-AutoRound

# --- M tier (medium) ---
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

# --- L tier (large / solo) ---
huggingface-cli download Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound

huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --local-dir $BASE/Nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

huggingface-cli download openai/gpt-oss-120b \
  --local-dir $BASE/OpenAI/GPT-OSS-120B
```

### GGUF models (llama.cpp)

```bash
BASE=$LLM_ROOT_PATH/ollama   # e.g. /home/YOUR_USER/LLMs/ollama

huggingface-cli download HauhauCS/Qwen3.5-35B-A3B-Uncensored-Aggressive \
  --include "*.gguf" --include "*.jinja" \
  --local-dir $BASE/Alibaba/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

---

## Step 7 — Configure llama-swap

Copy the full sample config (it contains all models pre-configured with correct paths):

```bash
cp llama-swap/config.yaml.sample llama-swap/config.yaml
```

Then do a global replace of `/path/to/` with your actual `LLM_ROOT_PATH`:

```bash
sed -i "s|/path/to/LLMs|$LLM_ROOT_PATH|g" llama-swap/config.yaml
sed -i "s|/path/to/Docker|$HOME/Docker|g" llama-swap/config.yaml
```

The sample covers all models in the tier table above. Key structural rules:
- `swap: false` keeps all group members loaded simultaneously.
- `exclusive: true` on the `large-models` group evicts S and M tiers automatically when any 120B+ model loads.
- Every `cmd` uses `--network container:llama-swap` — model containers share llama-swap's network namespace and bind to `localhost:${PORT}`.

See [llama-swap/config.yaml.sample](llama-swap/config.yaml.sample) for the complete annotated config.

---

## Step 8 — Configure LiteLLM

Copy the sample and update your master key:

```bash
cp LiteLLM/config.yaml.sample LiteLLM/config.yaml
sed -i "s|sk-your-litellm-master-key|$LITELLM_MASTER_KEY|g" LiteLLM/config.yaml
```

The sample wires every llama-swap model through `http://llama-swap:8080/v1` (Docker DNS) and sets per-model reasoning flags:

| Model | `supports_reasoning` | `merge_reasoning_content_in_choices` |
|---|---|---|
| Qwen3.5-35B-FP8 | true | true |
| Qwen3.5-122B-int4 | true | false (separate `reasoning` field) |
| Qwen3-Omni / Qwen3-Coder | true | true |
| Qwen3-VL | false | — |
| Nemotron-4B-FP8 | false | — |
| Nemotron-30B-NVFP4 | true | true |
| Nemotron-Super-120B | true | true |
| GPT-OSS-120B | true | true |
| Mistral-Small-24B | false | — |

See [LiteLLM/config.yaml.sample](LiteLLM/config.yaml.sample) for the complete annotated config.

---

## Step 9 — Dynamic launcher for large models (122B+)

For any model that occupies >80 GB, residual CUDA memory from the previous model can cause vLLM's startup sanity check to fail (`free_memory < gpu_memory_utilization × total`).

The included script queries `nvidia-smi` at launch time and computes a safe `--gpu-memory-utilization` dynamically:

```bash
# llama-swap/scripts/launch-qwen35-122b.sh
# Usage: /app/scripts/launch-qwen35-122b.sh ${PORT} ${host}
```

Reference it from `config.yaml`:

```yaml
  Qwen3.5-122B-A10B-int4-AutoRound:
    ttl: 3600
    readyTimeout: 1800
    cmd: /app/scripts/launch-qwen35-122b.sh ${PORT} ${host}
    cmdStop: "docker stop vllm-qwen3.5-122b-${PORT}"
```

---

## Step 10 — docker-compose.yml

Copy the sample and replace all `YOUR_USER` / `YOUR_*` placeholders:

```bash
cp docker-compose.yml.sample docker-compose.yml
```

Sanitized reference:

```yaml
version: '3.8'

services:
  vllm:
    image: ghcr.io/YOUR_GH_USER/vllm-spark:latest
    container_name: vllm
    restart: unless-stopped
    ipc: host
    ports: ["18000:8000"]
    volumes:
      - /home/YOUR_USER/LLMs/safetensors:/model
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    command: >
      --model /model/Qwen3.5-7B-Instruct --host 0.0.0.0 --port 8000
    networks: [dgx_net]

  llama-server:
    image: ghcr.io/YOUR_GH_USER/llama-cpp-spark:latest
    container_name: llama-cpp
    ports: ["19000:19000"]
    ulimits: { memlock: -1, stack: 67108864 }
    volumes:
      - /home/YOUR_USER/LLMs/llama:/models
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    command: --host 0.0.0.0 --port 19000 --models-dir /models --n-gpu-layers 99
    networks: [dgx_net]

  llama-swap:
    image: ghcr.io/YOUR_GH_USER/llama-swap-spark:latest
    container_name: llama-swap
    restart: unless-stopped
    ports: ["28080:8080"]
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
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    networks: [dgx_net]

  ollama:
    image: ghcr.io/YOUR_GH_USER/ollama-spark:latest
    container_name: ollama
    ports: ["11434:11434"]
    volumes:
      - /home/YOUR_USER/LLMs/ollama:/root/.ollama
    networks: [dgx_net]

  litellm-db:
    image: postgres:15-alpine
    container_name: litellm-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: litellm_admin
      POSTGRES_PASSWORD: YOUR_DB_PASSWORD
    ports: ["15432:5432"]
    volumes:
      - litellm_db_data:/var/lib/postgresql/data
    networks: [dgx_net]

  litellm:
    image: ghcr.io/YOUR_GH_USER/litellm-spark:latest
    container_name: litellm
    restart: unless-stopped
    depends_on: [litellm-db]
    ports: ["14000:4000"]
    environment:
      - DATABASE_URL=postgresql://litellm_admin:YOUR_DB_PASSWORD@litellm-db:5432/litellm
      - LITELLM_MASTER_KEY=YOUR_MASTER_KEY
    volumes:
      - /home/YOUR_USER/Docker/REPO_DIR/LiteLLM/config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    networks: [dgx_net]

networks:
  dgx_net:
    external: true

volumes:
  litellm_db_data:
```

---

## Step 11 — Verify

```bash
# Check llama-swap model list
curl http://localhost:28080/v1/models | python3 -m json.tool

# Check LiteLLM health
curl http://localhost:14000/health

# Watch llama-swap logs (shows model load/unload events)
docker logs -f llama-swap
```

---

## Step 12 — Benchmark

> **Credit: [@eugr](https://github.com/eugr) — [llama-benchy](https://github.com/eugr/llama-benchy)**
>
> Standardized LLM benchmark tool built for the Spark community. Output format is designed for direct copy-paste into forum posts for cross-system comparison. Thank you @eugr.

```bash
pip install llama-benchy
bash benchmark-models.sh --endpoint http://localhost:28080
```

---

## Tips & common issues

**Two models loading simultaneously → OOM**
Every group must be `swap: true` for solo models, or `exclusive: true` for L-tier. On 128 GB unified memory, even two 35B FP8 models at `gpu_mem=0.5` (64 GB each) can exceed safe limits. Tune per-model `gpu_memory_utilization` to fit your concurrent group math.

**vLLM startup check fails: `free_memory < gpu_memory_utilization × total`**
Use the dynamic launcher script (Step 7) for any L-tier model. It reads `nvidia-smi` at launch time and computes a safe utilization fraction. For M/S tier, `0.40` and `0.20` respectively are reliably safe.

**Mamba/hybrid models crash at startup**
Add `--mamba-ssm-cache-dtype float16` and use `vllm-node-tf5` (built with `--tf5`). The standard `vllm-node` image does not include transformers v5 patches required by Mamba hybrid architectures.

**GPT-OSS-120B: never use `--distributed-executor-backend ray` solo**
Ray's GCS server adds ~500 MB overhead and its 95% OOM monitor kills the worker before the model finishes loading. The default `mp` executor works fine. Ray is only useful for multi-node tensor parallelism.

**`--load-format fastsafetensors` cuts startup time ~40%**
Requires `model.safetensors.index.json` alongside the weight files. All HuggingFace multi-shard models include it. Avoid for models that load >85% of available RAM — fastsafetensors uses multi-threaded direct I/O which can cause OOM on unified memory if the model is very close to the limit.

**llama-swap shows "unhealthy" for a model that's loading**
Large models (122B+) need `readyTimeout: 1800` or more. The default 60 s is far too short for a 90 GB weight load.

**`--network container:llama-swap` vs. `network_mode: host`**
Model containers must use `--network container:llama-swap` so they share llama-swap's network namespace and bind to `localhost:${PORT}`. Do **not** use `host` networking for model containers — it breaks port isolation and llama-swap's health checks.

---

## Acknowledgements

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (pre-built GB10 vLLM wheels, nightly CI) and [llama-benchy](https://github.com/eugr/llama-benchy) (benchmarking). Both are essential to this stack. Thank you.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) (CUTLASS MXFP4 kernels for GB10 / GPT-OSS-120B). Thank you.
- **[mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)** — the VRAM orchestrator.
- **[BerriAI/litellm](https://github.com/BerriAI/litellm)** — the API gateway.
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)** and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — inference engines.

---

*Repo: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama*
