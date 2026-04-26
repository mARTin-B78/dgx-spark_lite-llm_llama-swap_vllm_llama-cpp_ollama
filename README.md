# DGX Spark AI Orchestration Stack

> **Full-stack LLM inference for NVIDIA DGX Spark (Grace-Blackwell GB10)**
> vLLM · llama.cpp · llama-swap · Ollama · LiteLLM — on 128 GB unified memory

> For a narrative walkthrough with benchmark numbers and deeper explanations, see [TUTORIAL.md](TUTORIAL.md).

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

## Getting Started

Choose one of two setup paths:

### 🚀 Option 1: Automated Setup (Recommended for First-Time Users)

**Time: ~5 minutes of guided prompts**

The interactive setup wizard handles everything automatically:

```bash
git clone https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama.git
cd dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama

# Run the interactive setup wizard
./setup/setup.sh

# Auto-download models based on your tier selections
./setup/download-models.sh

# Start the stack
docker compose up -d
```

**What the setup.sh script does:**
- ✅ Verifies Docker & NVIDIA Container Runtime installation
- ✅ Detects running services and available ports
- ✅ Guides through service selection (Portainer, LiteLLM, llama.cpp, Ollama, llama-swap)
- ✅ Collects credentials (HuggingFace token, GitHub PAT)
- ✅ Auto-resolves port conflicts with custom alternatives
- ✅ Lets you choose model tiers (S/M/L/GGUF)
- ✅ Auto-generates `.env` and `docker-compose.yml`
- ✅ Creates `SETUP_SUMMARY.txt` for reference

**What the download-models.sh script does:**
- Reads your model tier selections from `.env`
- Uses `hf download` for resumable, efficient transfers
- Filters GGUF models to Q4_K_M quantization (saves 70+ GB!)
- Shows progress and final disk usage

For full docs and troubleshooting, see [setup/README.md](setup/README.md).

---

### 📖 Option 2: Manual Setup (For Advanced Users / Full Control)

**Time: ~45 minutes of manual configuration**

For detailed step-by-step instructions with explanations, see [TUTORIAL.md](TUTORIAL.md).

#### Quick Manual Steps:

**Step 1 — Create the Docker network**

**Step 1 — Create the Docker network**

All containers share one bridge network so they can resolve each other by name (e.g. `http://llama-swap:8080`):

```bash
docker network create dgx_net
```

---

**Step 2 — Create `.env`

Copy the sample and fill in your values:

```bash
cp docker-compose.yml.sample docker-compose.yml
```

Edit the placeholder values:
- `<LLM_ROOT_PATH>` → Your model storage path (e.g., `/home/user/LLMs`)
- `<REPO_CONFIG_PATH>` → Repo root path (e.g., `/home/user/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama`)
- `<YOUR_GITHUB_PAT>` → Your GitHub Personal Access Token (optional for local builds)
- `<YOUR_POSTGRES_PASSWORD>` → Choose a secure database password
- `<YOUR_LITELLM_MASTER_KEY>` → Generate a secure API key (e.g., `sk-randomstring`)

For detailed guidance, see [docker-compose.yml.sample](docker-compose.yml.sample) header.

---

**Step 3 — Initialize Git submodules**

The `vllm/build/spark-vllm-docker` folder is a Git submodule. Initialize it:

```bash
git submodule update --init --recursive
cd vllm/build/spark-vllm-docker
```

---

**Step 4 — Build `vllm-node` (standard vLLM)

Used for: Qwen3.5-35B-FP8, Mistral-Small-24B, Nemotron-4B-FP8, Nemotron-30B-NVFP4, GPT-OSS-120B, Qwen3-VL, Qwen3-Omni.

```bash
./build-and-copy.sh
# Image tag defaults to: vllm-node
```

---

**Step 5 — Build `vllm-node-tf5` (transformers v5, Mamba/hybrid models)

Required for: Qwen3.5-122B-A10B-int4-AutoRound, Qwen3-Coder-Next, and any other hybrid Mamba architecture.

```bash
./build-and-copy.sh --tf5
# Image tag defaults to: vllm-node-tf5
```

---

**Step 6 — Build `vllm-node-mxfp4` (CUTLASS MXFP4 kernels)

Optional. Only needed for GPT-OSS-120B with native MXFP4 quantization (fastest path, ~60 tok/s).

```bash
./build-and-copy.sh --exp-mxfp4
# Image tag defaults to: vllm-node-mxfp4
```

---

**Step 7 — Download models

Install the HuggingFace CLI if you don't have it:

```bash
pip install huggingface-hub[cli]
```

Then download models using `hf download`. See [TUTORIAL.md Step 4](TUTORIAL.md#step-4-download-models) for the complete list of model commands.

Quick example:

```bash
BASE=$LLM_ROOT_PATH/vllm

# Small models (S tier)
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \
  --repo-type model --local-dir $BASE/Nvidia/Nemotron-3-Nano-4B-FP8

# Medium models (M tier)
hf download meta-llama/Llama-3.1-34B-Instruct \
  --repo-type model --local-dir $BASE/Meta/Llama-3.1-34B-Instruct

# Large models (L tier)
hf download meta-llama/Llama-3.3-70B-Instruct \
  --repo-type model --local-dir $BASE/Meta/Llama-3.3-70B-Instruct

# GGUF models (llama.cpp - Q4_K_M only)
hf download lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF \
  --repo-type model --include "*Q4_K_M*" \
  --local-dir $BASE/../gguf/Meta-Llama-3.1-70B-Instruct-GGUF
```

---

**Step 8 — Configure llama-swap

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

**Step 9 — Configure LiteLLM

Copy the sample and update your master key:

```bash
cp LiteLLM/config.yaml.sample LiteLLM/config.yaml
sed -i "s|sk-your-litellm-master-key|$LITELLM_MASTER_KEY|g" LiteLLM/config.yaml
```

The sample wires every llama-swap model through `http://llama-swap:8080/v1` (Docker DNS) and sets per-model reasoning flags.

---

**Step 10 — Start the stack

All containers are defined in `docker-compose.yml`. Start them:

```bash
docker compose up -d
```

Verify services are running:

```bash
docker compose ps
```

Expected output shows all services running (LiteLLM, llama-swap, Ollama, PostgreSQL, etc.).

---

**Step 11 — Test the API

```bash
# List available models
curl http://localhost:14000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# Make a test completion request
curl http://localhost:14000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "Nemotron-4B-FP8",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 100
  }'
```

---

**Step 12 — View logs

Monitor the stack in real-time:

```bash
docker compose logs -f

# Specific service logs
docker compose logs -f litellm
docker compose logs -f llama-swap
docker compose logs -f llama-cpp  # If installed
```

---

## Need More Details?

For the full narrative walkthrough with benchmark numbers and deeper explanations, see [TUTORIAL.md](TUTORIAL.md).

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

## Stack architecture reference

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

## Troubleshooting

### Setup script won't run
```bash
chmod +x setup/setup.sh
./setup/setup.sh
```

### Port already in use
The automated setup script auto-detects ports. If a port is taken, it will prompt for an alternative. For manual setup, edit `docker-compose.yml` port mappings.

### Docker daemon not running
```bash
sudo systemctl start docker
```

### NVIDIA Container Runtime not found
Check your `/etc/docker/daemon.json` includes:
```json
{
  "default-runtime": "nvidia"
}
```

Then restart Docker:
```bash
sudo systemctl restart docker
```

### Models not downloading
- Ensure `pip install huggingface-hub[cli]` is installed
- Verify `hf` command works: `hf --version`
- For faster transfers: `pip install hf-transfer` and set `export HF_HUB_ENABLE_HF_TRANSFER=1`

---

## Additional Resources

- **[TUTORIAL.md](TUTORIAL.md)** — Detailed step-by-step guide with benchmarks
- **[setup/README.md](setup/README.md)** — Setup wizard documentation and troubleshooting
- **[setup/setup.sh](setup/setup.sh)** — Interactive configuration wizard
- **[setup/download-models.sh](setup/download-models.sh)** — Automated model downloader
- **[docker-compose.yml.sample](docker-compose.yml.sample)** — Service definitions
- **[llama-swap/config.yaml.sample](llama-swap/config.yaml.sample)** — VRAM orchestrator config
- **[LiteLLM/config.yaml.sample](LiteLLM/config.yaml.sample)** — API gateway config

---

## Next Steps

1. **Choose your setup path:**
   - **Automated:** `./setup/setup.sh` (~5 min)
   - **Manual:** Follow the steps above or [TUTORIAL.md](TUTORIAL.md) (~45 min)

2. **Download models:**
   - **Automated:** `./setup/download-models.sh`
   - **Manual:** Use `hf download` commands from Step 7

3. **Start the stack:** `docker compose up -d`

4. **Verify:** `curl http://localhost:14000/v1/models -H "Authorization: Bearer YOUR_KEY"`

---

## What is this?

A production-ready Docker Compose stack that lets a single DGX Spark run **multiple large language models** without manual VRAM juggling. `llama-swap` acts as the orchestrator: it spins up the right inference container on demand and evicts it when idle, so 128 GB is never wasted on a model nobody is using.

A unified **LiteLLM** gateway exposes every model through one OpenAI-compatible endpoint with a single API key — no per-service port juggling.

---

## Stack overview

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
