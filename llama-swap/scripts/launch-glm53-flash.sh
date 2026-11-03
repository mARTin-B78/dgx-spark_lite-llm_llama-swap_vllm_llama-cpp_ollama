#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 2.05bpw + DFlash2 (vLLM, locally built gitcommit90/glm-5.3-one-spark @ddc28b1).
# Build: cd /home/sparky/Docker/glm-5.3-one-spark && docker build -t glm53-one-spark-vllm:tp1-general23 .
# Recipe default gpu-memory-utilization 0.90 starves the host on GB10. vLLM needs GMU x 121.7 GiB FREE in
# cudaMemGetInfo at startup (page cache is NOT counted: drop caches first); vLLM sees ~5-6 GiB less than a bare
# torch probe. Ladder by free CUDA memory (max GMU 0.86): >=103.5 GiB: 0.85/65536/1; >=99.5: 0.82/15360/1;
# >=97.5: 0.80/8192/4. Default 0.78/8192/1: the only config that survived a full tool-eval run on 2026-11-02
# (KV 11.65 GiB = 11,776 tokens, MemAvailable min ~3 GiB); 0.80/8192/4 was stopped by the 3 GiB guard under load.
# Knobs reach the container both via sed (pinned ddc28b1 serve script) and ONE_SPARK_* env (recipe PR #4 script).
# Garbage output on 2026-11-02 was 3 corrupt (sparse) weight shards, not the image: sha256-verify after download.
set -euo pipefail
port=${1:?port required}
host=${2:-0.0.0.0}
repo=/home/sparky/Docker/glm-5.3-one-spark
# Optional local overrides (GLM53_IMAGE/GMU/CTX/SEQS/CACHE...) without a llama-swap restart; runtime/ is not committed.
override="$(dirname "$0")/../runtime/glm53.env"
[[ -f $override ]] && . "$override"
gmu=${GLM53_GMU:-0.78}
ctx=${GLM53_CTX:-8192}
seqs=${GLM53_SEQS:-1}
models=/home/sparky/LLMs/glm53
cache=${GLM53_CACHE:-/home/sparky/.cache/glm53-one-spark}
image=${GLM53_IMAGE:-glm53-one-spark-vllm:tp1-general23}
# GLM at gmu>0.8 leaves <10 GiB host headroom by design; user-approved 3 GiB watchdog floor (default stays 10).
export GUARD_FLOOR_GIB=${GLM53_GUARD_FLOOR_GIB:-3}
exec bash "$(dirname "$0")/guard-large-model.sh" "vllm-glm53-flash-$port" 60 \
  --ipc=host --shm-size 32g --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e GLOO_SOCKET_IFNAME=lo -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e EXL3_FUSED_MOE=1 -e EXL3_MOE_ROW_TILE=0 -e EXL3_TEMP_ROWS_FUSED=128 \
  -e EXL3_FAT_SORTED=0 -e EXL3_FAT_BATCHED=0 -e EXL3_FAT_KERNEL=1 \
  -e GLM53_SUPPRESS_STOPS_IN_REASONING=1 -e GLM53_MIXED_PREFILL_CHUNK=skip \
  -e GLM53_INDEXER_WORKSPACE=stock -e GLM53_SPINWAIT_MS=stock \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e ONE_SPARK_HOST="$host" -e ONE_SPARK_PORT="$port" -e ONE_SPARK_K="${ONE_SPARK_K:-5}" \
  -e ONE_SPARK_UTIL="$gmu" -e ONE_SPARK_CTX="$ctx" -e ONE_SPARK_SEQS="$seqs" \
  -e ONE_SPARK_ASYNC="${ONE_SPARK_ASYNC:-}" -e ONE_SPARK_DRAFT_BLOCK="${ONE_SPARK_DRAFT_BLOCK:-}" \
  -e ONE_SPARK_MAMBA_SEED_FIX="${ONE_SPARK_MAMBA_SEED_FIX:-1}" \
  -v "$models/GLM-5.3-Flash-exl3-2.05bpw:/model:ro" \
  -v "$models/GLM-5.3-Flash-DFlash2:/draft:ro" \
  -v "$repo/scripts/serve-one-spark.sh:/start.sh:ro" \
  -v "$cache/vllm:/root/.cache/vllm" -v "$cache/triton:/root/.triton/cache" \
  -v "$cache/tilelang:/root/.tilelang/cache" -v "$cache/torchinductor:/tmp/torchinductor_root" \
  --entrypoint bash "$image" \
  -c 'sed -e "s/--gpu-memory-utilization 0.90/--gpu-memory-utilization '"$gmu"'/" -e "s/--max-model-len 262144/--max-model-len '"$ctx"'/" -e "s/--max-num-seqs 4/--max-num-seqs '"$seqs"'/" /start.sh > /tmp/s.sh && exec bash /tmp/s.sh'
