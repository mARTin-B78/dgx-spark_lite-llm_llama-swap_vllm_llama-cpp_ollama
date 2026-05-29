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
| **S** Small | 0.12–0.25 | 15–32 GB | 4× | Nemotron-4B-FP8, Qwen3.5-35B-GGUF, Qwen3.6-27B-heretic-GGUF |
| **M** Medium | 0.40–0.50 | 51–64 GB | 2× | Qwen3.5-35B-FP8, Qwen3.6-27B-PrismaSCOUT-NVFP4 |
| **L** Large | 0.65–0.85 | ~83–109 GB | 1× (solo) | Qwen3.6-35B-FP8, Qwen3.5-122B, Nemotron-120B, GPT-OSS-120B |

`exclusive: true` on the L-tier means loading any large model automatically evicts all others.

---

## Prerequisites

- NVIDIA DGX Spark (GB10) running Ubuntu 24.04
- Docker with NVIDIA Container Toolkit (`nvidia-ctk`)
- A container registry to host your images — **any** OCI-compatible registry works:
  - GitHub Container Registry (`ghcr.io`) — default
  - GitLab Container Registry (`registry.gitlab.com/<group>/<project>`)
  - Harbor, Nexus, AWS ECR, or any self-hosted registry
- A registry token/PAT with push permissions (`write:packages` for GHCR)

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

**Step 4 — Build `vllm-node` (standard vLLM)**

Used for: Qwen3.5-35B-FP8, Mistral-Small-24B, Nemotron-4B-FP8, Nemotron-30B-NVFP4, GPT-OSS-120B, Qwen3-VL, Qwen3-Omni.

```bash
./build-and-copy.sh
# Outputs: vllm-node:latest (matches llama-swap/config.yaml defaults perfectly, no manual retagging needed)
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

## Step 9 — Adaptive launcher (one launcher for all models)

vLLM's startup check fails (`free_memory < gpu_memory_utilization × total`) whenever a hardcoded `--gpu-memory-utilization` is too high for what's actually free at launch time. On unified-memory GB10 this happens often: residual CUDA memory from a previous model lingers, page cache holds prior weights, etc. The fix is to compute `--gpu-memory-utilization` adaptively at launch time.

[llama-swap/scripts/launch-vllm-auto.sh](llama-swap/scripts/launch-vllm-auto.sh) does this for any vLLM model. It estimates required VRAM from `weights (safetensors) + KV cache (config.json) + safety`, queries `/proc/meminfo`, applies a configurable system-RAM ceiling, and picks the smallest `--gpu-memory-utilization` that fits — or pins it to a static value if you supply one.

```yaml
  Qwen3.6-35B-A3B-FP8:
    ttl: 600
    readyTimeout: 600
    checkEndpoint: "/health"
    cmd: >
      env
      MODEL_PATH=/models/vllm/Alibaba/Qwen3.6-35B-A3B-FP8
      MODEL_HOST_PATH=/models/vllm/Alibaba/Qwen3.6-35B-A3B-FP8
      CONTAINER_NAME=vllm-qwen3.6-35b-${PORT}
      IMAGE=vllm-node-tf5:latest
      PORT=${PORT} HOST=${host}
      MAX_MODEL_LEN=131072 MAX_NUM_SEQS=10 KV_DTYPE_BYTES=1
      GMEM_MIN=0.55 GMEM_MAX=0.85 SAFETY_GIB=4
      /app/scripts/launch-vllm-auto.sh
      --served-model-name Qwen3.6-35B-A3B-FP8
      --kv-cache-dtype fp8
      --load-format fastsafetensors
      --attention-backend FLASHINFER
      --enable-prefix-caching
    cmdStop: "docker stop vllm-qwen3.6-35b-${PORT}"
```

### Launcher env reference

**Adaptive vs static — single knob:**

| Env | Default | Effect |
|---|---|---|
| `GMEM_OVERRIDE` | unset | Numeric value (e.g. `0.7069`) pins gpu_memory_utilization to that exact value, bypassing the calculation entirely. Unset, empty, or `adaptive` = compute dynamically. **One line to flip a model between hand-tuned and adaptive.** |

**Adaptive bounds (used when `GMEM_OVERRIDE` is unset):**

| Env | Default | Purpose |
|---|---|---|
| `GMEM_MIN` | `0.55` | Floor for the computed value. Launcher fails fast if this can't fit. |
| `GMEM_MAX` | `0.92` | Ceiling for the computed value. Use lower values for thermal- or workload-sensitive models. |
| `SAFETY_GIB` | `4` | Headroom added to the weights+KV estimate. |
| `KV_DTYPE_BYTES` | `1` | `1` for fp8, `2` for bf16/fp16. Drives KV-cache size estimate. |
| `MAX_MODEL_LEN` | `131072` | Context length used in KV estimate (passed to vllm too). |
| `MAX_NUM_SEQS` | `10` | Concurrent sequences used in KV estimate. |

**System-wide guards:**

| Env | Default | Purpose |
|---|---|---|
| `SYSTEM_RAM_CEILING_GIB` | `117.81` | Hard cap on total system RAM the launcher will plan against. Defaults to **126.5 GB decimal** because GB10 systems crash near this threshold. The launcher always reserves `(MemTotal - ceiling)` GiB, even when `/proc/meminfo` says more is available. |
| `GMEM_FREE_BUFFER_GIB` | `5` | Buffer between launcher's view of free RAM and CUDA's. `MemAvailable` and `cudaMemGetInfo` can disagree by ~1 GiB at vLLM startup (kernel reclaims page cache during `cudaMalloc`, not before the check), so the launcher plans against `MemAvailable - 5 GiB`. |
| `CUDA_OVERHEAD_GIB` | `6.3` | GiB of `MemTotal` not visible to CUDA on GB10 unified memory. |

**Image / container plumbing:**

| Env | Default | Purpose |
|---|---|---|
| `IMAGE` | required | Docker image (e.g. `vllm-node:latest`, `vllm-node-tf5:latest`). |
| `EXTRA_DOCKER_ARGS` | empty | Extra `docker run` args (mounts, envs). Example: `'-e VLLM_USE_FLASHINFER_MOE_FP8=1 -v /host/mod:/mod:ro'`. |
| `PRE_LAUNCH_CMD` | empty | Bash command run inside the container before `vllm serve`. Used for image patches (`bash run.sh`) or env setup. When set, the launcher uses `--entrypoint /bin/bash -c "PRE_LAUNCH_CMD && exec vllm serve …"`. |
| `VLLM_SERVE_PREFIX` | `vllm serve` | Set to `""` (empty) for images whose ENTRYPOINT is already `vllm serve`, so the model path and flags pass directly as entrypoint args. |

**A second launcher** — [llama-swap/scripts/launch-qwen35-122b.sh](llama-swap/scripts/launch-qwen35-122b.sh) — exists for historical reasons (uses `nvidia-smi` instead of `/proc/meminfo`). The generic launcher above can replace it; pick whichever you prefer.

---

## Step 10 — docker-compose.yml

Copy the sample:

```bash
cp docker-compose.yml.sample docker-compose.yml
```

All image references expand `${REGISTRY}` and `${IMAGE_NAMESPACE}` from `.env`, so no manual editing is needed — just make sure `setup.sh` (or your `.env`) has those values set. Example snippet:

```yaml
services:
  vllm:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_NAMESPACE:-${GH_USER}}/vllm-spark:latest
  llama-server:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_NAMESPACE:-${GH_USER}}/llama-cpp-spark:latest
  llama-swap:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_NAMESPACE:-${GH_USER}}/llama-swap-spark:latest
  ollama:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_NAMESPACE:-${GH_USER}}/ollama-spark:latest
  litellm:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_NAMESPACE:-${GH_USER}}/litellm-spark:latest
```

For non-ghcr.io registries, add to `.env`:

```env
REGISTRY=registry.gitlab.com
IMAGE_NAMESPACE=mygroup/myproject
REGISTRY_USER=myuser
REGISTRY_TOKEN=mytoken
```

Then run the helper to rewrite the inline image strings inside `llama-swap/config.yaml` (the only place env-var expansion doesn't reach automatically):

```bash
./setup/rewrite-registry.sh            # applies changes
./setup/rewrite-registry.sh --dry-run  # preview first
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

`benchmark-models.sh` wraps two complementary tools and runs them back-to-back against every model in your llama-swap instance — loading each model once and running both passes before unloading:

- **[llama-benchy](https://github.com/eugr/llama-benchy)** ([@eugr](https://github.com/eugr)) — standardized throughput benchmark (pp/tg/TTFT). Output is a markdown table designed for direct copy-paste into NVIDIA forum posts for cross-system comparison.
- **[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)** ([@SeraphimSerapis](https://github.com/SeraphimSerapis)) — 69-scenario tool-calling quality benchmark. Optional, via `--quality`. See the [forum announcement](https://forums.developer.nvidia.com/t/introducing-tool-eval-bench-cli/366903).

### Install prerequisites

```bash
pip install llama-benchy                                                    # speed benchmark
uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git  # quality benchmark (optional)
```

### Interactive wizard (no-args default)

Running the script with no arguments on an interactive terminal launches a guided setup. It fetches the live model list from llama-swap, lets you pick models and tests, choose a speed profile and quality mode, then shows the equivalent CLI command before running:

```bash
bash benchmark-models.sh          # launches the interactive wizard
```

To skip the wizard and go straight to CLI flags, pass any flag (e.g. `--quick`) or set `BENCH_NO_WIZARD=1`.

### Speed profiles

All profiles use pp=2048, tg=128 (real-world document workload shape). The `depth` values simulate KV-cache pressure as context fills up.

| Flag | Profile name | Depths | Runs | Use case |
|---|---|---|---|---|
| *(none)* | Medium Log | 0, 16384 | 3 | Default. ~50-page document. Baseline + one realistic context depth. |
| `--quick` | Smoke test | 0, 16384 | 1 | Fast sanity check after a config change. |
| `--stress` | Massive Log | 0, 16384, 32768 | 3 | Doubles context to ~100 pages. Reveals bandwidth bottleneck. |
| `--extreme` | Extreme Limit | 0, 16384, 32768, 65535 | 3 | ~200-page corpus. Crash/swap detection. |
| `--full` | Full sweep | 0, 16384, 32768 | 3 | Broad: pp=512+2048, tg=128+256+512. Detailed throughput map. |
| `--arena` | Spark-Arena | 0–100000 (7 depths) | 3 | Official [spark-arena.com](https://spark-arena.com) leaderboard profile. 4 concurrency levels. |

Examples:

```bash
bash benchmark-models.sh                          # Medium Log, all models
bash benchmark-models.sh --stress Qwen3.5 Nemotron  # Stress test, only matching models
bash benchmark-models.sh --quick                  # Smoke test, 1 run, fast
bash benchmark-models.sh --runs 5 Qwen3.6-35B    # Custom run count
```

### What the numbers mean

| Metric | What it measures | DGX Spark typical range |
|---|---|---|
| **pp tok/s** | Prompt Processing — how fast the model reads your input | 500–5000+ tok/s |
| **tg tok/s** | Token Generation — how fast the model writes its reply (what you "feel") | 15–60+ tok/s |
| **TTFT** | Time To First Token — delay before streaming starts | 100–2000 ms |

**The key insight for DGX Spark:** compare `tg tok/s` across depths. A large drop from `depth=16384` to `depth=32768` reveals the unified memory bandwidth bottleneck between the ARM CPU and the Blackwell GPU. The script automatically highlights this with a percentage and `⚠ BANDWIDTH BOTTLENECK DETECTED` if the drop exceeds 15%.

### Quality mode (tool-call accuracy)

`--quality` runs tool-eval-bench right after llama-benchy, on the same loaded model, so no second load is needed. Results land in `test-results/quality/` and are also stored in `data/benchmarks.sqlite` for structured querying.

```bash
bash benchmark-models.sh --quality                              # short: 15 core scenarios (~2-5 min/model)
bash benchmark-models.sh --quality --quality-mode full          # full: 69 scenarios (~15-30 min/model)
bash benchmark-models.sh --quality --quality-mode hardmode      # full + 5 adversarial scenarios
bash benchmark-models.sh --quality-only                         # skip speed, only measure quality
bash benchmark-models.sh --quality --quality-categories "K A J" # restrict to specific categories (A-P)
```

The summary table at the end of a run includes a `Quality /100` column with a star rating. Per-model markdown reports land in `test-results/quality/<YYYY>/<MM>/<run_id>.md`.

#### `data/benchmarks.sqlite`

tool-eval-bench writes every completed run to a local SQLite database at `data/benchmarks.sqlite`. The schema is:

```
scenario_runs (run_id, created_at, status, model, config_json, scores_json, metadata_json)
```

You can query it directly to compare models over time:

```bash
sqlite3 data/benchmarks.sqlite \
  "SELECT model, json_extract(scores_json,'$.score'), json_extract(scores_json,'$.rating')
   FROM scenario_runs WHERE status='completed' ORDER BY created_at DESC;"
```

The database is committed to the repo so benchmark history is tracked in git alongside config changes.

### Spark Arena leaderboard submission

`--arena` uses the official [spark-arena.com](https://spark-arena.com) benchmark profile (7 depths × 4 concurrency levels). Results and recipe files are saved to `test-results/arena-submission/<timestamp>/<model>/`. If the result is a new personal best, the script offers to submit via `spark-arena-cli` (auto-installed to `~/.local/bin` on first run):

```bash
bash benchmark-models.sh --arena                  # run arena profile, all models
spark-arena-cli login                              # authenticate (Google / GitHub)
spark-arena-cli benchmark test-results/arena-submission/<timestamp>/<model>/recipe.yaml
```

### Crash resume

If a long benchmark run is interrupted, `--resume` picks up from the last completed model:

```bash
bash benchmark-models.sh --stress --resume        # skip models already finished in last session
```

Checkpoint files are stored in `test-results/checkpoints/`.

### Results layout

```
test-results/
  benchmarks/            # llama-benchy — JSON + Markdown per model per run
  quality/               # tool-eval-bench — per-run markdown reports
  arena-submission/      # arena CSV + recipe.yaml (--arena mode)
  arena-best-results.json  # personal bests for leaderboard tracking
  checkpoints/           # crash-resume state
data/
  benchmarks.sqlite      # structured quality results (tool-eval-bench), queryable with sqlite3
```

---

## Tips & common issues

**Two models loading simultaneously → OOM**
Every group must be `swap: true` for solo models, or `exclusive: true` for L-tier. On 128 GB unified memory, even two 35B FP8 models at `gpu_mem=0.5` (64 GB each) can exceed safe limits. Tune per-model `gpu_memory_utilization` to fit your concurrent group math.

**vLLM startup check fails: `free_memory < gpu_memory_utilization × total`**
Switch the model block to [llama-swap/scripts/launch-vllm-auto.sh](llama-swap/scripts/launch-vllm-auto.sh) (see Step 9 for the env-var reference). It picks a fitting `--gpu-memory-utilization` from actually-free RAM and clamps to `[GMEM_MIN, GMEM_MAX]`. To pin a model to a static value instead, set `GMEM_OVERRIDE=0.7069` (or whichever number) — single env-var flip, no restructuring. To pin a static value across the whole host, set `SYSTEM_RAM_CEILING_GIB` (defaults to `117.81`, = 126.5 GB decimal — the observed crash threshold on GB10).

**System crashes at ~126.5 GB used RAM**
GB10 unified memory becomes unstable above this point. The launcher above defaults to a `SYSTEM_RAM_CEILING_GIB=117.81` cap so the gmem calculation always reserves `(MemTotal - 117.81)` GiB of headroom even when `/proc/meminfo` says more is available. Override per-model with `SYSTEM_RAM_CEILING_GIB=...` if your specific workload tolerates more pressure.

**Mamba/hybrid models crash at startup**
Add `--mamba-ssm-cache-dtype float16` and use `vllm-node-tf5` (built with `--tf5`). The standard `vllm-node` image does not include transformers v5 patches required by Mamba hybrid architectures.

**GPT-OSS-120B: never use `--distributed-executor-backend ray` solo**
Ray's GCS server adds ~500 MB overhead and its 95% OOM monitor kills the worker before the model finishes loading. The default `mp` executor works fine. Ray is only useful for multi-node tensor parallelism.

**`--load-format fastsafetensors` cuts startup time ~40%**
Requires `model.safetensors.index.json` alongside the weight files. All HuggingFace multi-shard models include it. Avoid for models that load >85% of available RAM — fastsafetensors uses multi-threaded direct I/O which can cause OOM on unified memory if the model is very close to the limit.

**Blank line in a `cmd: >` block causes silent OOM on hybrid models (Qwen3.6-27B)**
YAML folded block scalar (`>`) preserves blank lines as literal `\n`. A shell running the resulting command treats that newline as a statement separator: the `env VAR=val …` prefix becomes a no-op (env without a trailing command just prints the environment), and none of `MAX_MODEL_LEN` / `MAX_NUM_SEQS` / `GMEM_MAX` reach the launch script. For hybrid Qwen3.6 models this is fatal: vLLM auto-detects `max_model_len=262144` from `config.json` instead of the intended 65536, allocates 81 GiB KV cache, and crashes the system. **Never put a blank line between env-var assignments and the script call inside a `>` block.**

**llama-swap shows "unhealthy" for a model that's loading**
Large models (122B+) need `readyTimeout: 1800` or more. The default 60 s is far too short for a 90 GB weight load.

**`--network container:llama-swap` vs. `network_mode: host`**
Model containers must use `--network container:llama-swap` so they share llama-swap's network namespace and bind to `localhost:${PORT}`. Do **not** use `host` networking for model containers — it breaks port isolation and llama-swap's health checks.

---

## Acknowledgements

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (pre-built GB10 vLLM wheels, nightly CI) and [llama-benchy](https://github.com/eugr/llama-benchy) (benchmarking). Both are essential to this stack. Thank you.
- **[@SeraphimSerapis](https://github.com/SeraphimSerapis)** — [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench): 69-scenario tool-calling quality benchmark used by `benchmark-models.sh --quality`.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) (CUTLASS MXFP4 kernels for GB10 / GPT-OSS-120B). Thank you.
- **[mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)** — the VRAM orchestrator.
- **[BerriAI/litellm](https://github.com/BerriAI/litellm)** — the API gateway.
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)** and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — inference engines.

---

*Repo: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama*
