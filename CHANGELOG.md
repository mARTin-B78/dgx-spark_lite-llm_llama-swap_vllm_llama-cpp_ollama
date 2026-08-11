# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.11.1] — 2026-08-11

### Fixed
- **`benchmark-models.sh`**: replaced `eval` of hand-built command strings in
  `run_benchy`/`run_quality` with array-based invocations. A model name (or
  `--quality-extra-args` value) containing shell metacharacters could previously
  be interpreted as shell code instead of a literal argument.
- **`docker-compose.yml`/`.sample`**: restored the `${IMAGE_NAMESPACE:-${GH_USER}}`
  fallback on all image references — it was accidentally dropped to a bare
  `${IMAGE_NAMESPACE}` in 0.11.0, breaking image pulls for any `.env` that only
  sets `GH_USER`.

---

## [0.11.0] — 2026-08-11

### Added
- **`llama-qwen36-27b`**: second always-on persistent llama.cpp service (Qwen3.6-27B
  abliterated, Q4_K_M, ~16.5 GB) alongside the existing always-on 4B, exposed on
  `19002`. Stays resident regardless of what llama-swap loads/evicts.
- **`LiteLLM/complexity_hook.py`**: `COMPLEX`/`REASONING` tiers now route to the new
  always-on 27B instead of falling back to the 4B — the complexity router has real
  tiering again instead of collapsing everything onto one model. Also clamps
  client-supplied `temperature` into the `[0, 2]` OpenAI-compatible range instead of
  passing invalid values through.
- New `llama-swap` model entries: `Qwen3.6-35B-A3B-int4-AutoRound`,
  `Qwen3.6-35B-A3B-PrismaQuant-4.75bit`, `Qwen3.5-122B-A10B-NVFP4` (txn545 single-Spark
  MTP checkpoint, plus `llama-swap/scripts/patch-qwen35-nvfp4-runtime.sh` to register
  its custom model class before vLLM starts), and the AEON-7 DFlash/Multimodal-NVFP4-MTP
  variants (`Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash` /
  `-Multimodal-NVFP4-MTP`), all mirroring forum-recommended single-Spark recipes.
- **`sglang`** service in `docker-compose.yml` (scitrera DGX Spark image) as an
  additional serving backend alongside vLLM/llama.cpp/Ollama.
- **`LLM_OLLAMA_ROOT_PATH`**: optional dedicated host path for the Ollama model cache,
  independent of `LLM_ROOT_PATH`; defaults to `${LLM_ROOT_PATH}/ollama` if unset.
- **`benchmark-models.sh --size-order`**: sort matched models smallest-to-largest
  before running, plus a per-model RAM-budget lookup so overnight runs are easier to
  compare at a glance.
- `docker-compose.yml`/`.sample`: `logging.max-size`/`max-file` on the `llama-swap`
  service — the default json-file driver never rotated, and one oversized logged
  response (seen: a 7.5 MB `/api/events` entry) could corrupt the file for `docker
  logs`' full-history reader while `docker logs -f` kept working, making the log look
  silently "frozen".

### Changed
- **`overnight.sh`**: rewritten to run under `set -euo pipefail`, log each model's
  benchmark output to its own timestamped file under `test-results/overnight-logs/`
  in addition to a master log, and continue past a failed model instead of aborting
  the whole sequence.
- Default `ollama` service in `docker-compose.yml` commented out (superseded by the
  dedicated always-on llama.cpp services); `IMAGE_NAMESPACE` no longer silently falls
  back to `${GH_USER}` in image references, so a missing `IMAGE_NAMESPACE` now fails
  loudly instead of resolving to an unintended tag.
- `scripts/start-ds4-deepseek.sh`: DSpark speculative decoding now opt-in
  (`ENABLE_DSPARK=0` by default) — measured peak usage with DSpark enabled
  (~104–113 GiB against the ~121.69 GiB unified pool) reliably triggered real
  `NVRM: Out of memory` driver errors that could destabilize the whole graphical
  session, not just the process; without DSpark the model fits comfortably
  (~83–90 GiB). Also adds `--gpu-vram`/`--batched-session` args.

### Removed
- `Qwen3.6-27B-PrismaQuant-5.5bit` / `Qwen3.6-27B-uncensored-heretic-vllm` dropped
  from `parse_metrics.py` and `overnight.sh` in favor of
  `Qwen3.6-35B-A3B-PrismaQuant-4.75bit`.
- `Nemotron-3-Nano-30B-A3B-NVFP4` commented out of `llama-swap/config.yaml.sample`.

---

## [0.10.5] — 2026-08-04

### Added
- **`scripts/recipe_tool.py import-sparkrun`**: translate an upstream
  [sparkrun/spark-arena recipe](https://sparkrun.dev/recipes/format/)
  (`model`/`runtime`/`container`/`defaults`/`command` with `{placeholder}`
  substitution) directly into a `llama-swap` `config.yaml` model block —
  `{port}`/`{host}` map to llama-swap's own `${PORT}`/`${host}` launch macros
  rather than the recipe's literal defaults, every other placeholder bakes in
  from `defaults`, and the container is wrapped with this stack's usual
  `--runtime nvidia --gpus all --ipc=host --network container:llama-swap`
  plus an HF cache mount (sparkrun recipes assume the runtime downloads the
  model by HF id on first launch). Supports `vllm` and `llama-cpp` runtimes —
  the two this stack already knows how to run; other runtimes are rejected
  with a clear error rather than producing something broken. Same dry-run/
  `--apply`/auto-backup behavior as `import`.

---

## [0.10.4] — 2026-08-04

### Added
- **`scripts/recipe_tool.py`** & **`scripts/recipe.sh`**: recipe import/export for
  `llama-swap/config.yaml` model blocks. `recipe.sh export <model>` writes a model's
  `cmd`/`ttl`/`checkEndpoint`/`proxy`/etc. block to a standalone, versionable file
  under `recipes/`; `recipe.sh import <recipe-file>` applies it back into
  `config.yaml` (dry-run diff by default, `--apply` to write, auto-backs up the
  config first). Round-trips comments and block-scalar (`>`) formatting elsewhere
  in the file via `ruamel.yaml`, so importing/exporting a single model doesn't
  disturb the rest of a hand-maintained config.
- **`llama-swap/scripts/lib-wait-mem-stable.sh`**: shared `wait_for_stable_memory`
  helper, sourced by all three vLLM launch scripts (`launch-vllm-auto.sh`,
  `launch-qwen35-122b.sh`, `launch-qwen35-122b-hybrid.sh`). Polls `MemAvailable`
  until two consecutive readings agree, guarding against a launch reading a
  `/proc/meminfo` snapshot taken mid-teardown of the previous model's container
  (docker/CUDA context release is async, so an immediate read can undercount
  free memory or race an in-flight unload).

---

## [0.10.3] — 2026-07-06

### Changed
- **`Qwen3.5-122B-A10B-int4-AutoRound`**: Added DFlash speculative decoding (`z-lab/Qwen3.5-122B-A10B-DFlash`, 15 spec tokens) to the launch script for massive throughput gains.
- **`Qwen3.5-27B-Uncensored-DFlash-NVFP4`**: Updated attention backend to `flashinfer` in `llama-swap/config.yaml` to improve agentic tool-calling accuracy.
- **`Qwen3.6-27B-PrismaSCOUT-NVFP4`** & **`Qwen3.6-27B-uncensored-heretic-vllm`**: Updated attention backend to `flashinfer` to maximize quality across NVFP4 models.
- **`overnight.sh`**: Expanded script to cover all 20 registered models and added `--quality-mode full` for rigorous 69-scenario `tool-eval-bench` validation.

### Tooling
- Upgraded `tool-eval-bench` CLI to v2.1.0 to enforce strict agentic fact extraction in Hard Mode.

---

## [0.10.2] — 2026-06-25

### Changed
- **`Qwen3.5-122B-A10B-int4-AutoRound`**: Optimized launch script (`launch-qwen35-122b.sh`) for single-user interactive latency by reducing `--max-num-seqs` (10 → 3) and `--max-num-batched-tokens` (32768 → 8192).
- **`Nemotron-3-Nano-30B-A3B-NVFP4`**: Optimized prefill performance in `config.yaml.sample` to prevent 180s HTTP timeouts on large prompts. Halved context to 131k, reduced concurrency to 4, enabled chunked prefill, and added `expandable_segments` PyTorch alloc config to prevent unified-memory swap thrashing.

---

## [0.10.1] — 2026-06-17

### Removed
- **`Qwen3.5-122B-A10B-hybrid-int4fp8`** model registration removed from
  `llama-swap/config.yaml.sample` and `benchmark-models.sh`. The hybrid checkpoint
  loads successfully but generates garbled/incoherent output (confirmed even at
  temperature=0, with and without the `mods/fix-qwen3.5-hybrid-int4fp8` patches), so it
  is no longer exposed as a selectable model. The `mods/` and launch script remain in
  place for future work once the checkpoint is rebuilt. Use
  `Qwen3.5-122B-A10B-int4-AutoRound` instead.

---

## [0.10.0] — 2026-06-12

### Added
- **`mods/fix-qwen3.5-hybrid-int4fp8/`** *(spark-vllm-docker)*: New mod applying three
  stacked optimizations to the Qwen3.5-122B-A10B-int4-AutoRound inference path, lifting
  throughput from 28.3 → ~51 tok/s (+80%) on a single DGX Spark:
  - `patch_inc.py`: Patches vLLM's INC quantization backend to detect FP8 dense layers
    in a hybrid checkpoint and dispatch them through CUTLASS block-wise FP8 GEMM
    (`Fp8LinearMethod`) instead of the BF16 fallback (+8.8%). Works with vLLM 0.21+.
  - `patch_int8_lmhead.py`: INT8 LM Head v2 — replaces the per-token Python loop with a
    single batched 2D Triton GEMV kernel with `@triton.autotune` for SM121 (+~40%).
  - `host/build-hybrid-checkpoint.py`: One-time host-side script that merges MoE expert
    weights (INT4, from Intel AutoRound) with dense layer weights (FP8 E4M3, from the
    official Qwen/Qwen3.5-122B-A10B-FP8 checkpoint) into a single hybrid safetensors
    checkpoint (~9 GB smaller than the INT4 original).
  - `host/add-mtp-weights.py`: Surfaces the MTP speculative-decoding tensors already
    present in the AutoRound checkpoint by adding them to the index of the hybrid
    checkpoint, enabling MTP-2 (`num_speculative_tokens=2`, ~80% accept rate, +25%).
- **`recipes/qwen3.5-122b-hybrid-int4fp8.yaml`** *(spark-vllm-docker)*: Recipe for the
  hybrid model; includes the new mod, sets `--speculative-config mtp:2`, single-GPU,
  `--kv-cache-dtype fp8`, `--attention-backend FLASHINFER`.
- **`llama-swap/scripts/launch-qwen35-122b-hybrid.sh`**: llama-swap launch script for the
  hybrid checkpoint with adaptive `gpu_memory_utilization`, MTP-2 speculative decoding,
  and inline mod application at container start.

---

## [0.9.0] — 2026-06-12

### Added
- **`docker-compose.yml`**: New `llama-qwen35-4b` always-on service — dedicated persistent
  llama.cpp instance for Qwen3.5-4B-Q4_K_M; stays loaded regardless of llama-swap evictions,
  sized for the STT→LLM→TTS pipeline (low latency, always warm, ctx 131072).
- **`LiteLLM/complexity_hook.py`**: Custom `CustomLogger` pre-call hook that rewrites `model`
  to a complexity-tiered target before LiteLLM routes the request. Fires on all endpoints
  (including `/v1/responses`) where the native `auto_router` type is unsupported.
- **`LiteLLM/router.json`**: Semantic router config for embedding-based intent routing
  (requires `OPENAI_API_KEY` for the embedding call).
- **`scripts/start-ds4-deepseek.sh`**: Startup script for the DS4 DeepSeek node.

### Changed
- **`LiteLLM/config.yaml.sample`**: Add `auto_router1` (complexity-based, 4 tiers: SIMPLE →
  4B, MEDIUM → 27B, COMPLEX → 35B, REASONING → 122B) and `semantic-router` (embedding-based
  intent routing) model entries.
- **`docker-compose.yml`** / **`docker-compose.yml.sample`**: Mount `router.json` and
  `complexity_hook.py` into the LiteLLM container; add optional `OPENAI_API_KEY` env var.

### Chore
- **`.gitignore`**: Ignore `ds4/` (compiled binaries + model), `logs/`, `*.o`, `*.pid`.

---

## [0.8.0] — 2026-06-12

### Fixed
- **`launch-vllm-auto.sh`**: Broken `/models/vllm` volume mount when `LLM_ROOT_PATH` already
  points to the vllm directory (e.g. `/home/user/LLMs/vllm`). Script was appending `/vllm`,
  producing a double-`vllm` path that Docker auto-created as an empty directory — all vllm
  models started with an empty `/models/vllm` mount and failed with "chat template not found".
  Default fallback updated from `/home/user/LLMs` → `/home/user/LLMs/vllm`.

### Changed
- **`llama-swap/config.yaml.sample`**: Set all model `ttl: 0` (was `ttl: 3600`/`600`/`300`).
  Models now stay loaded in VRAM until the swap mechanism evicts them when a different model
  is requested, instead of auto-unloading after an idle timeout.

### Changed *(config.yaml — gitignored, applied manually)*
- **Qwen3.6-35B-A3B-FP8**: Removed `GMEM_OVERRIDE=0.7069`; model now uses adaptive
  `gpu_memory_utilization` based on free memory at launch time.
- **Qwen3.6-35B-A3B-FP8**: Fixed `MODEL_HOST_PATH` from `/models/vllm/Alibaba/…` →
  `/models/Alibaba/…` so the adaptive launcher can read `config.json` inside the llama-swap
  container (where `LLM_ROOT_PATH` is mounted as `/models`).
- **Qwen3.6-35B-A3B-FP8**: Lowered `GMEM_MIN` `0.55` → `0.40` so adaptive does not abort
  when TTS services are running and effective free memory yields u_cap ≈ 0.44.

---

## [0.7.0] — 2026-05-29

### Fixed
- `docker-compose.yml`: Comment out static vllm sidecar service; fix llama-server port to 19000
- `docker-compose.yml`: Pass `LLM_ROOT_PATH` into llama-swap container env so launch scripts
  can build correct host-side volume paths
- `.env.sample`: Add missing `REGISTRY` and `IMAGE_NAMESPACE` variables
- `.env.sample`: Make `POSTGRES_DB` and `POSTGRES_USER` configurable (were hardcoded)
- `docker-compose.yml.sample`: Fully sync structure and comments with live config

### Changed
- `docker-compose.yml`: Align live compose structure with sample for easier diffing

### Added
- Qwen3.6-27B-PrismaSCOUT-NVFP4 and Qwen3.6-27B variants to `config.yaml.sample`

### Docs
- `benchmark-models.sh`: Document usage in README
- Standardize vllm-node image tags across all model blocks; add German non-technical explanation

---

## [0.6.0] — 2026-05-18

### Security
- Remove hardcoded credentials and absolute paths from `docker-compose.yml`

### Fixed
- Qwen3.6-35B-A3B-Uncensored: Disable thinking mode; expand context to 64K
- Qwen3.6-27B: Resolve OOM crash at startup

### Docs
- Sync `docker-compose.yml.sample` with live stack configuration (secrets scrubbed)
- `config.yaml.sample`: Clarify two vllm image families (vllm-node vs vllm/vllm-openai)
- README: Add private-registry support documentation

---

## [0.5.0] — 2026-05-06

### Added
- Private registry support: `REGISTRY` and `IMAGE_NAMESPACE` env vars for GitLab, Harbor,
  Nexus, and other non-ghcr.io registries
- `benchmark-models.sh`: Interactive wizard, quality detail report, robust model unload
- `benchmark-models.sh`: Tool-eval-bench integration for tool-call quality scoring
- `benchmark-models.sh`: `--arena` mode, coherence detection, spark-arena-cli integration
- `benchmark-models.sh`: S/M/L concurrent request groups and `--resume`

### Fixed
- Issues #6, #7, #8 in setup and build scripts

### Docs
- Expand launcher reference: `GMEM_OVERRIDE`, system RAM ceiling, environment variable plumbing

---

## [0.4.0] — 2026-05-03

### Added
- **`launch-vllm-auto.sh`**: `GMEM_OVERRIDE` knob — numeric value pins `gpu_memory_utilization`
  statically; `"adaptive"` / unset computes dynamically from free memory
- 126.5 GB system RAM ceiling (`SYSTEM_RAM_CEILING_GIB`) to prevent GB10 unified-memory crash
  when total system RAM approaches the hardware limit
- 5 GiB `u_cap` buffer (`GMEM_FREE_BUFFER_GIB`) to bridge `MemAvailable` vs `cudaMemGetInfo`
  discrepancy at vLLM startup
- `VLLM_SERVE_PREFIX` env var for images whose entrypoint is already `vllm serve`
  (e.g. `vllm/vllm-openai`)
- `PRE_LAUNCH_CMD` env var for in-container patching or setup before `vllm serve`

### Fixed
- Launcher: Use `/proc/meminfo` instead of `nvidia-smi` for memory queries (GB10 compatibility)
- Launcher: Shell-only implementation (awk/sed/grep) — no Python in the minimal llama-swap image
- Adaptive gmem for mod-script models; corrected pp display in output

---

## [0.3.0] — 2026-04-28

### Added
- **`launch-vllm-auto.sh`**: Generic adaptive `--gpu-memory-utilization` for vLLM — estimates
  required VRAM from safetensor weights + KV cache + safety headroom, picks the smallest
  utilization that satisfies the estimate within `[GMEM_MIN, GMEM_MAX]`

### Fixed
- `llama-swap` dynamic VRAM allocation for 122B model; fix docker-compose networking
- `benchmark-models.sh`: Support llama-benchy installed via pip in addition to uvx
- `gpu_memory_utilization` floor for 122B raised `0.60` → `0.82`

---

## [0.2.0] — 2026-04-19

### Added
- Sample configs for LiteLLM and llama-swap (sanitized)
- `benchmark-models.sh`: S/M/L concurrent groups, `--resume`, `--arena` mode,
  coherence detection, spark-arena-cli integration
- `tool-eval-bench` runner; tuned 122B launcher for tool calling
- README, TUTORIAL.md with setup guide, benchmarks, and model download commands

### Fixed
- `benchmark-models.sh`: Stabilize script; enhance coherence checks

---

## [0.1.0] — 2026-03-29

### Added
- Unified DGX Spark / Grace-Blackwell AI orchestration stack
- llama-swap orchestrator for on-demand model loading/eviction
- vLLM support for safetensors models (FP8, NVFP4, compressed-tensors)
- llama.cpp support for GGUF models
- Ollama support for pulled models via modelfile format
- LiteLLM unified API gateway on port 14000
- GB10 unified-memory optimizations across all services
- Docker publish workflow for multiple image builds

---

<!-- version diff links — update tags in GitHub after each release -->
[Unreleased]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/releases/tag/v0.1.0
