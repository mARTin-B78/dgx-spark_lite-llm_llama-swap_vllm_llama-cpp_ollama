# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **launch-vllm-auto.sh**: Fixed broken `/models/vllm` volume mount when `LLM_ROOT_PATH` already points to the vllm directory (e.g. `/home/user/LLMs/vllm`). The script was appending `/vllm` to `LLM_ROOT_PATH`, producing a double-`vllm` path (`…/vllm/vllm`) that Docker auto-created as an empty directory — causing all vllm models to see an empty `/models/vllm` mount and fail with "chat template not found". Default fallback value updated to `/home/user/LLMs/vllm` to match the convention.

---

## [9 commits since initial stack] — 2026-05-29

### Fixed
- `docker-compose.yml`: Comment out static vllm service; fix llama-server port to 19000
- `docker-compose.yml`: Pass `LLM_ROOT_PATH` into llama-swap container environment so launch scripts can construct correct host-side volume paths
- `.env.sample`: Add missing `REGISTRY` and `IMAGE_NAMESPACE`; make `POSTGRES_DB` and `POSTGRES_USER` configurable
- `docker-compose.yml.sample`: Sync with live config (secrets scrubbed)
- Qwen3.6-27B: Resolve OOM crash; add both 27B variants to sample config

### Docs
- Benchmark script: Document `benchmark-models.sh` usage in README
- vllm-node image tags: Standardize across all model blocks; add German non-technical explanation
- `config.yaml.sample`: Clarify two image families

---

## Launcher — adaptive gpu_memory_utilization

### Added / Fixed (launcher)
- `launch-vllm-auto.sh`: Generic adaptive `--gpu-memory-utilization` based on `/proc/meminfo` (works on GB10 unified memory where `nvidia-smi` memory queries return "Not Supported")
- `GMEM_OVERRIDE` knob: Numeric value pins utilization statically; `"adaptive"` / unset computes dynamically
- 126.5 GB system RAM ceiling to prevent GB10 unified-memory crash near full allocation
- 5 GiB `u_cap` buffer to bridge `MemAvailable` vs `cudaMemGetInfo` discrepancy at vLLM startup
- `VLLM_SERVE_PREFIX` for images whose entrypoint is already `vllm serve`
- Shell-only implementation (awk/sed/grep) — no Python dependency in the minimal llama-swap container

---

## Benchmark

### Added
- `benchmark-models.sh`: Interactive wizard, quality detail report, robust unload
- Tool-eval-bench integration for tool-call quality scoring
- `--arena` mode, coherence detection, spark-arena-cli integration
- S/M/L concurrent groups and `--resume`

---

## Initial stack

### Added
- Unified DGX Spark / Grace-Blackwell stack: llama-swap + vLLM + llama.cpp + Ollama + LiteLLM
- GB10 unified-memory optimizations across all services
- Private registry support (GitLab, Harbor, Nexus)
- Setup wizard (`setup/setup.sh`) with NVIDIA runtime check
