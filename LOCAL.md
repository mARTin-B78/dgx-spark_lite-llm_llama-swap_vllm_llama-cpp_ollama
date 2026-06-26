# DGX Spark LLM Stack — Local Build Setup Guide

> **Repo location:** `$HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama`
>
> **Images stay local — nothing is pushed to any registry.**
>
> Type every command one at a time, press Enter, and wait for it to finish before typing the next.

---

## How it all fits together

```
You → LiteLLM :14000 → llama-swap :28080 → vLLM / llama.cpp container (loads on demand)
```

- **LiteLLM** is the front door. Every app (Open-WebUI, curl, etc.) talks to it on port 14000.
- **llama-swap** is the manager. When a model is requested it spawns the right container, serves it, then kills it after it has been idle for a set time — freeing VRAM for the next model.
- **vLLM / llama.cpp containers** are the actual AI engines, one per model, started and stopped on demand.
- All containers share the `dgx_net` Docker bridge network so they can reach each other by name.

---

## Step 1 — Enter the repo and create your settings file

```bash
cd $HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

Copy the sample settings file:

```bash
cp .env.sample .env
```

Open it for editing:

```bash
nano .env
```

Make the following changes. Use the arrow keys to navigate, type to replace, then **Ctrl+O** to save and **Ctrl+X** to exit.

| Find this line | Replace with |
|---|---|
| `REGISTRY=ghcr.io` | `REGISTRY=local` |
| `IMAGE_NAMESPACE=martin-b78` | `IMAGE_NAMESPACE=spark` |
| `GH_USER=your-github-username` | `GH_USER=$USER` |
| `GITHUB_TOKEN=ghp_your-token-here` | `GITHUB_TOKEN=unused` |
| `LLM_ROOT_PATH=/home/user/LLMs` | `LLM_ROOT_PATH=$HOME/models` |
| `REPO_CONFIG_PATH=/home/user/Docker/...` | `REPO_CONFIG_PATH=$HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama` |
| `LITELLM_MASTER_KEY=sk-choose-a-secure-key` | `LITELLM_MASTER_KEY=sk-mysecretkey123` *(invent your own)* |
| `POSTGRES_PASSWORD=choose-a-db-password` | `POSTGRES_PASSWORD=mydbpassword` *(invent your own)* |
| `DATABASE_URL="postgresql://litellm_admin:choose-a-db-password@..."` | `DATABASE_URL="postgresql://litellm_admin:mydbpassword@litellm-db:5432/litellm"` *(same password as above)* |
| `LITELLM_UI_PASSWORD=choose-a-ui-password` | `LITELLM_UI_PASSWORD=myuipassword` *(invent your own)* |

> `REGISTRY=local` and `IMAGE_NAMESPACE=spark` means Docker will look for images named `local/spark/...` — exactly what we build in Step 2. No internet registry is involved.

---

## Step 2 — Build the five base Docker images

Think of this as baking recipes into frozen meals. Each `docker build` command reads a Dockerfile (the recipe) and produces a Docker image (the frozen meal) stored locally. **5–15 minutes per image** — wait for each to finish before running the next.

Make sure you are in the repo folder:

```bash
cd $HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

**llama.cpp** — runs GGUF-format models, compiled specifically for the GB10 GPU (SM12.1):

```bash
docker build -t local/spark/llama-cpp-spark:latest -f llama-cpp/llama-cpp.Dockerfile llama-cpp/
```

**llama-swap** — the VRAM manager that loads and unloads models on demand:

```bash
docker build -t local/spark/llama-swap-spark:latest -f llama-swap/llama-swap.Dockerfile llama-swap/
```

**Ollama** — a user-friendly model runner included as an alternative engine:

```bash
docker build -t local/spark/ollama-spark:latest -f ollama/ollama.Dockerfile ollama/
```

**LiteLLM** — the unified API gateway (one endpoint for all models):

```bash
docker build -t local/spark/litellm-spark:latest -f LiteLLM/litellm.Dockerfile LiteLLM/
```

**vLLM wrapper** — lightweight base image; the real per-model vLLM images are built in Step 3:

```bash
docker build -t local/spark/vllm-spark:latest -f vllm/vllm.Dockerfile vllm/
```

Verify all five were created:

```bash
docker images | grep "local/spark"
```

You should see five rows, one for each image above.

---

## Step 3 — Build the vLLM model-serving images

These are the containers that vLLM actually runs inside — one spawned per model request by llama-swap. They come from [@eugr's spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) submodule, which ships pre-built wheels compiled for CUDA 13.1 / SM12.1 / ARM64 SBSA. Without them you would need to compile vLLM and FlashInfer from source (2–4 hours).

Make sure you are back in the repo root (not inside a subdirectory from Step 2):

```bash
cd $HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

The vLLM build system lives in [@eugr's spark-vllm-docker repo](https://github.com/eugr/spark-vllm-docker). The tutorial describes it as a git submodule, but this copy of the repo has no `.gitmodules` file so the submodule was never registered. Clone it directly instead:

```bash
mkdir -p vllm/build
git clone https://github.com/eugr/spark-vllm-docker vllm/build/spark-vllm-docker
```

Enter the build folder:

```bash
cd vllm/build/spark-vllm-docker
```

**Standard image** — used for Nemotron-30B, Qwen3-VL, Qwen3-Omni, Mistral, Nemotron-4B. Takes ~15 minutes with pre-built wheels:

```bash
bash build-and-copy.sh
```

**Transformers 5.x image** — required for Qwen3.6-35B, Qwen3-Coder-Next, Qwen3.5-122B, and any model using Mamba/hybrid architecture:

```bash
bash build-and-copy.sh --tf5
```

**[Optional] MXFP4 experimental image** — only needed for GPT-OSS-120B. Compiles FlashInfer and a CUTLASS fork from source (~1 hour). Credit: [@christopherowen](https://github.com/christopherowen):

```bash
bash build-and-copy.sh --exp-mxfp4
```

Return to the repo root:

```bash
cd ../../..
```

Confirm the images exist:

```bash
docker images | grep vllm-node
```

You should see `vllm-node` and `vllm-node-tf5` (plus `vllm-node-mxfp4` if you ran the optional build).

> **Two image families explained:**
> `local/spark/...` images (Step 2) are the long-running services — llama-swap, LiteLLM, Ollama, etc.
> `vllm-node` images (this step) are the ephemeral containers that llama-swap spawns per model and kills when idle.

---

## Step 4 — Create the model folders and download models

Create the directories:

```bash
mkdir -p $HOME/models/vllm
mkdir -p $HOME/models/ollama
```

Check that the `hf` CLI is available (it is already installed on this machine):

```bash
hf --version
```

If for any reason it is missing: `pip install huggingface-hub[cli]`

Log in if any models require a HuggingFace account (gated models):

```bash
hf auth login
```

*(Get a free token at huggingface.co → Settings → Access Tokens)*

Set a path shortcut:

```bash
BASE=$HOME/models/vllm
```

> Models are large — 10 GB to 90 GB each. Start with one S-tier model to verify the stack works before downloading everything.

### S tier — Small / fast

```bash
hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-4B-FP8

hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --local-dir $BASE/Nvidia/Nemotron-3-Nano-30B-A3B-NVFP4
```

### M tier — Medium (30B FP8)

```bash
hf download Qwen/Qwen3.5-35B-A3B-FP8 \
  --local-dir $BASE/Alibaba/Qwen3.5-35B-A3B-FP8

hf download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --local-dir $BASE/Alibaba/Qwen3-VL-30B-A3B-Instruct-FP8

hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir $BASE/Alibaba/Qwen3-Omni-30B-A3B-Instruct

hf download unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-FP8-Dynamic

hf download Intel/Qwen3-Coder-Next-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3-Coder-Next-int4-AutoRound
```

### L tier — Large / solo only (120B MoE — evicts all other models)

```bash
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --local-dir $BASE/Nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

hf download Intel/Qwen3.5-122B-A10B-int4-AutoRound \
  --local-dir $BASE/Alibaba/Qwen3.5-122B-A10B-int4-AutoRound

hf download openai/gpt-oss-120b \
  --local-dir $BASE/OpenAI/GPT-OSS-120B
```

### GGUF model (runs via llama.cpp)

The `--include` flag downloads only the Q4_K_M quantisation variant instead of every variant (which would be 100+ GB):

```bash
hf download HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive \
  --include "*Q4_K_M*" --include "*.jinja" \
  --local-dir $HOME/models/ollama/Alibaba/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
```

---

## Step 5 — Configure llama-swap and LiteLLM

Make sure you are in the repo folder:

```bash
cd $HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

### llama-swap config

Copy the template:

```bash
cp llama-swap/config.yaml.sample llama-swap/config.yaml
```

Replace path placeholders in one go:

```bash
sed -i 's|<LLM_ROOT_PATH>|$HOME/models|g' llama-swap/config.yaml
sed -i 's|/path/to/LLMs|$HOME/models|g' llama-swap/config.yaml
sed -i 's|/home/sparky/LLMs|$HOME/models|g' llama-swap/config.yaml
sed -i 's|/home/sparky|$HOME|g' llama-swap/config.yaml
```

Open the file and scroll through it to catch any remaining placeholders:

```bash
nano llama-swap/config.yaml
```

Use **Ctrl+W** to search for `sparky`, `<LLM`, or `/path/to` and replace anything left over.

> **`MODEL_PATH` vs `MODEL_HOST_PATH` explained:**
> Inside the llama-swap container your models are mounted at `/models` (because `LLM_ROOT_PATH` on the host maps there). So a model at `$HOME/models/vllm/Nvidia/Nemotron-3-Nano-4B-FP8` on your machine is `/models/vllm/Nvidia/Nemotron-3-Nano-4B-FP8` inside the container.
> - `MODEL_PATH` = the *inside-container* path (starts with `/models/`)
> - `MODEL_HOST_PATH` = the *host* path (starts with `$HOME/models/`) — used by the launch script to read the model's `config.json` before starting the sub-container

### LiteLLM config

Copy the template:

```bash
cp LiteLLM/config.yaml.sample LiteLLM/config.yaml
```

Inject your master key (replace `sk-mysecretkey123` with whatever you set in `.env`):

```bash
sed -i 's|sk-your-litellm-master-key|sk-mysecretkey123|g' LiteLLM/config.yaml
```

Open and verify:

```bash
nano LiteLLM/config.yaml
```

---

## Step 6 — Create the shared Docker network

All containers need to be on the same private network so they can find each other by name. You only ever do this once:

```bash
docker network create dgx_net
```

If you see `network with name dgx_net already exists` — that is fine, just continue.

---

## Step 7 — Create docker-compose.yml and start the stack

Copy the sample:

```bash
cp docker-compose.yml.sample docker-compose.yml
```

Docker Compose reads your `.env` automatically. Because you set `REGISTRY=local` and `IMAGE_NAMESPACE=spark`, it will look for images named `local/spark/llama-cpp-spark:latest` etc. — exactly what you built in Step 2.

Do a quick sanity check on your settings before launching:

```bash
grep -E "REGISTRY|IMAGE_NAMESPACE|LLM_ROOT_PATH|REPO_CONFIG_PATH" .env
```

Expected output:
```
REGISTRY=local
IMAGE_NAMESPACE=spark
LLM_ROOT_PATH=$HOME/models
REPO_CONFIG_PATH=$HOME/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama
```

Launch everything in the background:

```bash
docker compose up -d
```

Watch the startup logs (press **Ctrl+C** to stop watching — the containers keep running):

```bash
docker compose logs -f
```

Check all containers are running:

```bash
docker ps
```

You should see: `litellm`, `litellm-postgres`, `llama-swap`, `llama.cpp`, `ollama`.

---

## Step 8 — Verify the stack is working

**List all models llama-swap knows about:**

```bash
curl http://localhost:28080/v1/models | python3 -m json.tool
```

**Check LiteLLM health:**

```bash
curl http://localhost:14000/health
```

**Send a real test message** (swap in any model name you downloaded):

```bash
curl http://localhost:28080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Nemotron-3-Nano-4B-FP8","messages":[{"role":"user","content":"Say hello!"}]}'
```

The very first request for a model takes 30–120 seconds while weights load into GPU memory. Watch it happen:

```bash
docker logs -f llama-swap
```

After the idle timeout (600 seconds by default) llama-swap automatically stops the model container to free VRAM for the next request.

**LiteLLM web UI** — open in your browser:

```
http://localhost:14000/ui
```

Username: `admin` (or whatever you set for `LITELLM_UI_USERNAME`)
Password: whatever you set for `LITELLM_UI_PASSWORD`

---

## Common problems and fixes

**"Cannot connect to the Docker daemon"**

Add your user to the docker group, then log out and back in:

```bash
sudo usermod -aG docker $USER
```

**"Image not found: local/spark/llama-cpp-spark:latest"**

The build in Step 2 failed or used a different tag. Check what you have:

```bash
docker images | grep local
```

Compare the names to what docker-compose expects, then rebuild the missing one.

**vLLM crashes at startup: "free memory < gpu_memory_utilization × total"**

This happens when residual CUDA memory from the previous model hasn't been released yet, or the static utilization value is too high. The `launch-vllm-auto.sh` script in `llama-swap/config.yaml` handles this automatically by computing utilization from actual free RAM. Make sure your config.yaml model blocks use the `env MODEL_PATH=... /app/scripts/launch-vllm-auto.sh` pattern rather than a static `docker run --gpu-memory-utilization 0.7`.

**System crashes or becomes unstable near 126 GB RAM used**

The GB10's unified memory becomes unstable above ~126.5 GB. The auto-launcher defaults `SYSTEM_RAM_CEILING_GIB=117.81` so it always reserves a safe buffer. If you are using static `docker run` commands instead, lower `--gpu-memory-utilization`.

**Mamba/hybrid models (Qwen3.6, Qwen3-Coder-Next, Qwen3.5-122B) fail to load**

These require the `vllm-node-tf5` image (built with `bash build-and-copy.sh --tf5`) and the `--mamba-ssm-cache-dtype float16` flag in the vllm serve command. Check your `llama-swap/config.yaml` block for those models.

**Port already in use on startup**

Something on your machine is already using one of the exposed ports (14000, 28080, 11434, 19000). Find the conflict:

```bash
sudo ss -tlnp | grep -E "14000|28080|11434|19000"
```

Either stop the conflicting service, or change the left-hand port number in `docker-compose.yml` (e.g. change `14000:4000` to `14001:4000`).

**Container exits immediately after starting**

Read its logs:

```bash
docker logs <container-name>
```

Common causes: wrong path in config, model files not downloaded yet, or a YAML syntax error in `config.yaml`.

---

## Quick reference

```bash
# Start the stack
docker compose up -d

# Stop the stack (data is preserved, containers removed)
docker compose down

# Restart a single service after editing its config
docker compose restart litellm
docker compose restart llama-swap

# See what is running
docker ps

# Live logs from all services
docker compose logs -f

# Live logs from llama-swap only (watch model loads/unloads)
docker logs -f llama-swap

# List all locally built images
docker images | grep -E "local/spark|vllm-node"

# List models llama-swap knows about
curl http://localhost:28080/v1/models | python3 -m json.tool

# Send a test prompt via llama-swap directly
curl http://localhost:28080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Nemotron-3-Nano-4B-FP8","messages":[{"role":"user","content":"Hello!"}]}'

# Send a test prompt via LiteLLM (add your master key header)
curl http://localhost:14000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-mysecretkey123" \
  -d '{"model":"Nemotron-3-Nano-4B-FP8","messages":[{"role":"user","content":"Hello!"}]}'
```
