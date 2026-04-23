# Running a Full Multi-Model LLM Stack on DGX Spark (GB10) — With VRAM Orchestration

**Hardware:** NVIDIA DGX Spark (Grace-Blackwell GB10, 128 GB unified memory)
**Stack:** vLLM · llama.cpp · llama-swap · Ollama · LiteLLM

---

## The problem this solves

The DGX Spark has 128 GB of unified CPU/GPU memory — enough to fit several large models *individually*, but not all at once. The naive approach is to pick one model and leave it running. The problem: a 35B FP8 model takes ~51 GB. You've got 77 GB sitting idle whenever you switch tasks.

What I wanted was:
- Switch between a fast 4B for quick tasks, a 35B for general use, and a 122B for deep reasoning
- Never manually stop and restart containers
- One API endpoint for everything, compatible with Claude Code / Open WebUI / any OpenAI client
- No VRAM wasted when a model is idle

The solution is **llama-swap** as a VRAM orchestrator. It acts like a model router: you request a model by name, llama-swap spins up the right Docker container, routes your request, and tears it down after it's been idle for a configurable TTL. LiteLLM sits in front as the unified gateway.

---

## How the stack fits together

```
Your client (Claude Code / Open WebUI / curl)
          │
          ▼  port 14000  (OpenAI-compatible, single API key)
    ┌───────────┐
    │  LiteLLM  │  ← unified gateway, auth, model aliases
    └─────┬─────┘
          │  routes by model name
          ▼  port 28080
    ┌───────────┐
    │ llama-swap│  ← VRAM orchestrator (has /var/run/docker.sock)
    └─────┬─────┘
          │  spawns/kills containers on demand
    ┌─────┼────────────────────┐
    ▼     ▼                    ▼
  vLLM  llama.cpp           Ollama
  :PORT  :PORT              :11434
 (ephemeral, per model)
```

---

## Model tiers and VRAM math

128 GB total − ~20 GB OS/driver overhead = **~108 GB usable for models**.

I split models into three tiers in llama-swap's `groups` config:

| Tier | `gpu_memory_utilization` | VRAM each | Max concurrent | Use case |
|---|---|---|---|---|
| **S** Small | 0.12–0.22 | 15–28 GB | up to 4× | Quick tasks, fast chat |
| **M** Medium | 0.40 | ~51 GB | up to 2× | General reasoning, coding |
| **L** Large | 0.70–0.85 | ~90–109 GB | 1× only | Deep reasoning, 120B+ |

The L-tier group has `exclusive: true`, which tells llama-swap to evict all S and M models before loading a 120B. The S and M tiers have `exclusive: false` so they can coexist freely within their VRAM budget.

---

## Benchmark results (2026-04-22, single Spark GB10)

Measured with [llama-benchy](https://github.com/eugr/llama-benchy) by [@eugr](https://github.com/eugr).
- `pp2048` = prompt processing, 2048 tokens (higher is better)
- `tg128` = token generation, 128 tokens (higher is better)
- `@ d16384` = same tests with 16k context depth

### S tier — Small / fast

| Model | Engine | pp2048 (tok/s) | tg128 (tok/s) | Notes |
|---|---|---|---|---|
| Nemotron-3-Nano-4B-FP8 | vLLM | 8179 | 39.8 | Instant responder, great for orchestration |
| Nemotron-3-Nano-30B-A3B-NVFP4 | vLLM | 7417 | 55.9 | Fastest 30B on the Spark |
| Qwen3.5-35B-A3B-Uncensored-Q4_K_M | llama.cpp | 1798 | 57.1 | GGUF on llama.cpp, competitive generation speed |

### M tier — Medium

| Model | Engine | pp2048 (tok/s) | tg128 (tok/s) | Notes |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | vLLM | 4439 | 49.1 | Solid all-rounder with native reasoning |
| Qwen3.6-35B-A3B-FP8 | vLLM | 4969 | 49.5 | Slightly faster prefill than 3.5 |
| Qwen3-VL-30B-A3B-Instruct-FP8 | vLLM | 9217 | 51.9 | Vision model, exceptional prefill |
| Qwen3-Omni-30B-A3B-Instruct | vLLM | 5227 | 30.1 | Audio + image + text multimodal |
| Qwen3-Coder-Next-FP8-Dynamic | vLLM | 3946 | 32.9 | Full-precision coder |
| Qwen3-Coder-Next-int4-AutoRound | vLLM | 4425 | **66.7** | INT4 quant, fastest generation in M tier |

### L tier — Large (solo only)

| Model | Engine | pp2048 (tok/s) | tg128 (tok/s) | Notes |
|---|---|---|---|---|
| Qwen3.5-122B-A10B-int4-AutoRound | vLLM (tf5) | 2048 | 23.8 | Best reasoning, fits in ~90 GB |
| GPT-OSS-120B (MXFP4) | vLLM (mxfp4) | 4703 | **56.4** | Remarkable speed for a 120B model |
| Nemotron-3-Super-120B-A12B-NVFP4 | vLLM | 1823 | 14.5 | NVIDIA's flagship reasoning model |

**GPT-OSS-120B** at 56 tok/s is the standout — that's a 120B model generating tokens faster than most 35B models thanks to CUTLASS MXFP4 kernels on the Blackwell architecture.

---

## Three vLLM images, not one

Most models use the standard `vllm-node` image. Two model families need special builds:

| Image | Build flag | Required for |
|---|---|---|
| `vllm-node` | *(default)* | Most models |
| `vllm-node-tf5` | `--tf5` | Mamba/hybrid models (Qwen3.5-122B, Qwen3-Coder-Next) |
| `vllm-node-mxfp4` | `--exp-mxfp4` | GPT-OSS-120B with CUTLASS MXFP4 |

The `tf5` build includes transformers v5 patches required by hybrid Mamba/Transformer architectures. Without it, those models crash at startup with a missing attention layer error.

The `mxfp4` build is a separate CUTLASS kernel set from [@christopherowen](https://github.com/christopherowen/spark-vllm-mxfp4-docker) — it's what enables GPT-OSS-120B's exceptional speed.

All three builds are from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) by [@eugr](https://github.com/eugr), which ships nightly-tested GB10 wheels so you don't have to compile from source.

---

## The dynamic VRAM launcher (why 122B needs special handling)

The GB10 uses unified memory — the OS and GPU share the same physical pool. After a 90 GB model unloads, the OS doesn't immediately reclaim all its pages. When vLLM starts next and runs its pre-flight check (`free_memory >= gpu_memory_utilization × total`), it can fail because residual CUDA context from the previous container is still occupying memory.

The fix is a small launcher script that queries `nvidia-smi` at startup and computes a dynamically-adjusted utilization fraction:

```bash
# Simplified logic
FREE=$(nvidia-smi --query-gpu=memory.free --format=noheaders,nounits)
TOTAL_CUDA=124610   # what CUDA sees (not the full 131072 the OS reports)
SAFETY=3072
GMEM=$(awk "BEGIN { u=($FREE - $SAFETY) / $TOTAL_CUDA; if(u>0.85) u=0.85; if(u<0.60) u=0.60; printf \"%.2f\", u }")
exec docker run ... vllm serve ... --gpu-memory-utilization $GMEM
```

This adds about 3 seconds of overhead (docker run to query nvidia-smi inside an existing image) but eliminates all false-start failures on the 122B model. See [llama-swap/scripts/launch-qwen35-122b.sh](llama-swap/scripts/launch-qwen35-122b.sh) for the full script.

---

## Key vLLM flags explained

A few flags that make a meaningful difference on GB10:

**`--load-format fastsafetensors`** — Uses multi-threaded direct I/O instead of mmap. Cuts model load time by ~40% on Spark where mmap performance is poor. Requires `model.safetensors.index.json` to be present (all HF multi-shard models have it).

**`--attention-backend FLASHINFER`** — FlashInfer's GB10-tuned attention kernels. Noticeably faster than the Triton fallback for most models.

**`--kv-cache-dtype fp8`** — Halves KV cache memory usage. Allows significantly longer effective context before running out of VRAM.

**`--enable-prefix-caching`** — Caches the KV state of repeated prompt prefixes. Useful for system prompts and multi-turn conversations.

**`--mamba-ssm-cache-dtype float16`** — Required for any Mamba/hybrid model. Without it, the SSM state cache defaults to float32 and the model uses ~15% more VRAM.

**`--max-cudagraph-capture-size`** — For large MoE models, reduces the number of CUDA graph sizes captured at startup. Dramatically cuts cold-start time for 120B+ models at the cost of a small runtime overhead for unusual batch sizes.

---

## llama-swap config structure

The full sample is at [llama-swap/config.yaml.sample](llama-swap/config.yaml.sample). The critical parts:

```yaml
# Global timeouts — 1 hour matches a long reasoning session
timeout: 3600
readyTimeout: 3600
healthCheckTimeout: 900

groups:
  small-models:
    swap: false       # keep all members loaded simultaneously
    exclusive: false  # can coexist with medium-models group
    members:
      - Nemotron-3-Nano-4B-FP8
      - Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M-GGUF

  medium-models:
    swap: false
    exclusive: false
    members:
      - Qwen3.5-35B-A3B-FP8
      - Qwen3-VL-30B-A3B-Instruct-FP8
      - Qwen3-Omni-30B-A3B-Instruct
      - Qwen3-Coder-Next-FP8-Dynamic
      - Mistral-Small-24B-Instruct-2501

  large-models:
    swap: true        # only one large model at a time
    exclusive: true   # evicts S and M tiers when loading
    members:
      - Qwen3.5-122B-A10B-int4-AutoRound
      - Nemotron-3-Super-120B-A12B-NVFP4
      - GPT-OSS-120B

models:
  Qwen3.5-35B-A3B-FP8:
    ttl: 600          # unload after 10 min of inactivity
    cmd: >
      docker run --rm --name vllm-qwen3.5-35b-${PORT}
      --runtime nvidia --gpus all --ipc=host
      --network container:llama-swap   # ← share llama-swap's network namespace
      -e NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1
      -v /home/YOUR_USER/LLMs/vllm:/models/vllm
      vllm-node
      vllm serve /models/vllm/Alibaba/Qwen3.5-35B-A3B-FP8
      --served-model-name Qwen3.5-35B-A3B-FP8
      --host ${host} --port ${PORT}
      --gpu-memory-utilization 0.40
      --max-model-len 131072
      --kv-cache-dtype fp8
      --load-format fastsafetensors
      --attention-backend FLASHINFER
      --enable-prefix-caching
      --enable-auto-tool-choice
      --tool-call-parser qwen3_xml
      --reasoning-parser qwen3
    cmdStop: "docker stop vllm-qwen3.5-35b-${PORT}"
```

The `--network container:llama-swap` flag is the linchpin — it puts the model container inside llama-swap's network namespace so it binds to `localhost:${PORT}`, which llama-swap then proxies. Without it, llama-swap can't reach the model.

---

## LiteLLM reasoning config

Most Qwen and Nemotron models expose a `<think>` block in their output. LiteLLM needs to know about it so it strips or merges it correctly for clients that don't expect it:

```yaml
# LiteLLM/config.yaml.sample (abridged)
- model_name: Qwen3.5-35B-A3B-FP8
  litellm_params:
    model: openai/Qwen3.5-35B-A3B-FP8
    api_base: "http://llama-swap:8080/v1"
    api_key: "sk-your-key"
    supports_reasoning: true
    include_reasoning: true
    merge_reasoning_content_in_choices: true  # folds <think> into the text stream
    extra_body:
      enable_thinking: true
      chat_template_kwargs: {"enable_thinking": true}
```

`merge_reasoning_content_in_choices: false` (used for 122B) keeps the `reasoning_content` field separate — useful if your client can display the chain-of-thought panel separately.

See [LiteLLM/config.yaml.sample](LiteLLM/config.yaml.sample) for all models.

---

## Setup summary (abbreviated)

Full step-by-step guide is in [README.md](README.md). Quick version:

```bash
# 1. Network
docker network create dgx_net

# 2. Environment
cp .env.sample .env && nano .env   # fill in GH_USER, LLM_ROOT_PATH, keys

# 3. Build vLLM images (from vllm/build/spark-vllm-docker/)
./build-and-copy.sh              # → vllm-node
./build-and-copy.sh --tf5        # → vllm-node-tf5
./build-and-copy.sh --exp-mxfp4  # → vllm-node-mxfp4

# 4. Download models
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-35B-A3B-FP8 \
  --local-dir $LLM_ROOT_PATH/vllm/Alibaba/Qwen3.5-35B-A3B-FP8
# ... (see README for all models)

# 5. Config files
cp llama-swap/config.yaml.sample llama-swap/config.yaml
sed -i "s|/path/to/LLMs|$LLM_ROOT_PATH|g" llama-swap/config.yaml
cp LiteLLM/config.yaml.sample LiteLLM/config.yaml

# 6. Start the stack
docker compose up -d

# 7. Verify
curl http://localhost:28080/v1/models | python3 -m json.tool
curl http://localhost:14000/health
```

---

## Common issues

**vLLM startup fails with `free_memory < gpu_memory_utilization × total`**
Another model container didn't fully release CUDA memory yet. Use the dynamic launcher script for 120B+ models — it reads actual free VRAM at launch and adjusts the fraction accordingly.

**Model container starts but llama-swap reports unhealthy**
Check `readyTimeout`. The default is too short for large models. Set `readyTimeout: 1800` for anything 50B+.

**Mamba/hybrid model crashes immediately**
You're using `vllm-node` instead of `vllm-node-tf5`. Rebuild with `./build-and-copy.sh --tf5`.

**GPT-OSS-120B with `--distributed-executor-backend ray` crashes (solo node)**
Ray's GCS server adds ~500 MB overhead and its 95% OOM watchdog kills the vLLM worker. Remove the flag — the default `mp` executor works perfectly on a single Spark.

**Two S-tier models load fine, adding a third causes OOM**
Each vLLM instance pre-allocates `gpu_memory_utilization × 128 GB` at startup, even if you're not using all of it. Add up your group's utilization fractions and make sure they leave room for the OS. `3 × 0.20 = 0.60 × 128 GB = 77 GB` is fine. `4 × 0.25 = 1.0 × 128 GB` is not.

---

## Repo

Everything described here is in a single repo with sample configs, the dynamic launcher script, and the benchmark runner:

**https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama**

Sample configs are sanitized (no credentials, no local paths) and ready to copy.

---

## Credits

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (GB10-optimized vLLM builds, nightly CI) and [llama-benchy](https://github.com/eugr/llama-benchy) (benchmarks). This stack wouldn't exist without the pre-built wheels. Thank you.
- **[@christopherowen](https://github.com/christopherowen)** — [spark-vllm-mxfp4-docker](https://github.com/christopherowen/spark-vllm-mxfp4-docker) (CUTLASS MXFP4 build enabling GPT-OSS-120B at 56 tok/s). Thank you.
- **[mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)** — the VRAM orchestrator that makes all of this possible.
- **[BerriAI/litellm](https://github.com/BerriAI/litellm)** — unified API gateway.
