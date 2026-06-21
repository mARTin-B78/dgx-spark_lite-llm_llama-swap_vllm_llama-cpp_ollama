# DGX Spark LLM Stack - Setup Options

This folder contains tools and documentation for setting up the LLM orchestration stack. You have two setup approaches to choose from:

---

## 🚀 Option 1: Automated Setup (Recommended for First-Time Users)

**Use this if:** You want a guided, interactive setup that handles configuration, port detection, and file generation.

```bash
cd setup/
./setup.sh
```

**What it does:**
- ✅ Checks Docker installation and NVIDIA Container Runtime
- ✅ Detects running services and port availability
- ✅ Guides you through service selection (Portainer, LiteLLM, llama-swap, llama.cpp, Ollama, vLLM)
- ✅ Collects credentials (HuggingFace token, GitHub PAT)
- ✅ Lets you select model tiers (S/M/L/GGUF)
- ✅ Auto-generates `.env` and `docker-compose.yml`
- ✅ Creates `SETUP_SUMMARY.txt` for reference

**Time:** ~5 minutes of interactive prompts

**Output files:**
- `.env` - All paths, ports, and credentials
- `docker-compose.yml` - Pre-configured service definitions
- `setup/SETUP_SUMMARY.txt` - Your configuration reference

### Next steps

#### Review the generated files before proceeding
```bash
cat ../.env
cat ../docker-compose.yml
```

#### Copy and configure the llama-swap model config
```bash
cp ../llama-swap/config.yaml.sample ../llama-swap/config.yaml
```

#### Change directory to the parent directory
```bash
cd ..
```

#### Build the ephemeral vLLM images from the submodule (one-time, ~30-60 min)
```bash
git submodule update --init --recursive
cd vllm/build/spark-vllm-docker
./build-and-copy.sh              # vllm-node:latest         (most models)
./build-and-copy.sh --tf5        # vllm-node-tf5:latest     (Mamba/hybrid models)
./build-and-copy.sh --exp-mxfp4  # vllm-node-mxfp4:latest   (GPT-OSS-120B MXFP4, optional)
cd ../../..
```

#### Build and push the five stack images to your registry
```bash
./build_and_push.sh
```

#### Auto-download models based on your tier selections
```bash
./setup/download-models.sh
```

#### Login to your docker registry
Skip if you just ran ./build_and_push.sh — you are already logged in
```bash
docker login <registry> -u <username>  # Docker will prompt for your password / personal access token
```

#### Start the stack (images are now in the registry and can be pulled)
```bash

docker compose up -d
```

---

## 📖 Option 2: Manual Setup (For Advanced Users / Full Control)

**Use this if:** You prefer to understand and control every step, or need to customize beyond what the automated script offers.

**Steps:**
1. Read [../TUTORIAL.md](../TUTORIAL.md) - Comprehensive step-by-step guide
2. Manually edit `docker-compose.yml.sample` and rename to `docker-compose.yml`
3. Create and configure `.env` file
4. Set up model directories: `$HOME/LLMs/{vllm,ollama}`
5. Download models using provided commands
6. Start services with `docker compose up -d`

**Time:** ~45 minutes (depending on familiarity)

**When to use:**
- Need custom configurations not covered by the wizard
- Want to understand each component deeply
- Prefer script-free manual configuration

---

## 🔄 After Setup: Download Models

Whether you use automated or manual setup, you'll need to download models:

### Automated Download (Recommended)

```bash
./download-models.sh
```

This script reads your model tier selections from `.env` and downloads the appropriate models using `hf download`.

### Manual Download

See the [../TUTORIAL.md](../TUTORIAL.md#step-4-download-models) for individual `hf download` commands for each model.

---

## ⚙️ Understanding the Stack Architecture

```
Client (localhost:14000)
    ↓
LiteLLM Gateway (port 4000 internal, 14000 host)
    ↓
llama-swap VRAM Orchestrator (port 8080 internal, 28080 host)
    ├→ vLLM Server (for large models)
    ├→ llama.cpp (for GGUF models)
    └→ Ollama (for model management)
```

**Key Components:**
- **LiteLLM**: Unified OpenAI-compatible API endpoint
- **llama-swap**: VRAM manager for seamless multi-model switching
- **vLLM**: High-throughput inference engine for large models (spawned as ephemeral containers by llama-swap)
- **llama.cpp**: Optimized GGUF quantized model inference (spawned as ephemeral containers by llama-swap)
- **Ollama**: Built-in model management and API (spawned as ephemeral containers by llama-swap)

> **Note:** `vllm`, `llama-server`, and `ollama` are **disabled by default** in `docker-compose.yml` as persistent Compose services. llama-swap manages them as ephemeral containers instead, which lets it reclaim VRAM when models are idle. Running them as persistent services alongside llama-swap permanently occupies VRAM that llama-swap cannot free, and on a 128 GB GB10 this can cause out-of-memory crashes when loading large models. To enable a persistent service, add its profile name to `COMPOSE_PROFILES` in `.env`.

---

## 🆘 Troubleshooting

### Setup script won't run
```bash
# Make executable
chmod +x setup/setup.sh
./setup/setup.sh
```

### Port already in use
The setup script auto-detects ports. If a port is taken, it will prompt for an alternative.

### Docker daemon not running
```bash
sudo systemctl start docker
# or
sudo dockerd  # if running standalone
```

### NVIDIA Container Runtime not found
Check your `/etc/docker/daemon.json` includes:
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime"
    }
  }
}
```

Then restart Docker:
```bash
sudo systemctl restart docker
```

### Models not downloading
- Ensure HuggingFace token is set: `export HF_TOKEN=hf_...`
- Check internet connection
- Verify model names are correct (use `hf_transfer` for speed):
  ```bash
  pip install hf_transfer
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download model/name
  ```

---

## 📚 Additional Resources

- **[TUTORIAL.md](../TUTORIAL.md)** - Comprehensive manual setup guide
- **[docker-compose.yml.sample](../docker-compose.yml.sample)** - Service definitions template
- **[llama-swap/](../llama-swap/)** - VRAM orchestrator configs
- **[LiteLLM/](../LiteLLM/)** - API gateway configs
- **[vllm/](../vllm/)** - Large model serving configs

---

## 🚀 Quick Start After Setup

```bash
# 1. Review configuration
cat .env
cat docker-compose.yml

# 2. Download models (if not already done)
./setup/download-models.sh

# 3. (vLLM models only, one-time) Build ephemeral inference images
#    → See "Build the ephemeral vLLM images from the submodule" in Option 1 above

# 4. Start the stack
docker compose up -d

# 5. Check services
docker compose ps

# 6. Test the API
curl http://localhost:14000/v1/models

# 7. View logs
docker compose logs -f litellm
docker compose logs -f llama-swap
```

---

## 💡 Tips

- **Keep your credentials safe:** Never commit `.env` to git
- **Review before deploying:** Always check generated files before running `docker compose up`
- **Port flexibility:** All ports are configurable if defaults conflict
- **Model storage:** Ensure enough disk space (~100+ GB for full tier setup)
- **Monitor VRAM:** Check `docker compose logs llama-swap` to see model loading

---

Generated: 2026-04-26
Updated: 2026-06-16
For issues or questions: [GitHub Issues](https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/issues)
