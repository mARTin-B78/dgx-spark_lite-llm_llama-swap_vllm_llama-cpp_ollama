#!/usr/bin/env bash
# Huihui FP8: one 64K request; 50 GiB available includes startup reserve.
# Checkpoint revision: badcb1961b646b3bdf1fc9ebdd9727826a55a48f
set -euo pipefail
port=${1:?port required}
host=${2:-127.0.0.1}
priority_args=(--disable-priority-preemption)
[[ ${QWEN38_PRIORITY_PREEMPTION:-0} != 1 ]] || priority_args=()
exec bash "$(dirname "$0")/guard-large-model.sh" "qwen38-huihui-fp8-$port" 50 \
  --ipc host --shm-size 2g --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  -e MAX_JOBS=4 -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor -e TRITON_CACHE_DIR=/root/.cache/triton \
  -v /home/sparky/LLMs/qwen38/huihui-fp8-runtime-cache:/root/.cache \
  -v /home/sparky/LLMs/qwen38/munekazu/Huihui-Qwen3.8-27B-abliterated-FP8:/models/qwen38:ro \
  sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3 \
  python3 -c '
from pathlib import Path
import runpy
path = Path("/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py")
source = path.read_text()
before = "        for rid in rids:\n            self.rid_to_state.pop(rid, None)"
after = "        for rid in rids:\n            try:\n                self.abort_request(rid)\n            except Exception:\n                logger.exception(\"Failed to cancel discarded request\")\n            self.rid_to_state.pop(rid, None)"
if source.count(before) != 1:
    raise RuntimeError("Pinned SGLang cancellation patch no longer matches")
patched = source.replace(before, after)
compile(patched, str(path), "exec")
path.write_text(patched)
runpy.run_module("sglang.launch_server", run_name="__main__")
' \
  --model-path /models/qwen38 --served-model-name Huihui-Qwen3.8-27B-abliterated-FP8 \
  --host "$host" --port "$port" \
  --quantization fp8 --dtype bfloat16 \
  --attention-backend flashinfer --kv-cache-dtype auto \
  --mem-fraction-static 0.55 --chunked-prefill-size 2048 \
  --context-length 65536 --max-total-tokens 65536 \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 1 --max-mamba-cache-size 8 --cuda-graph-max-bs-decode 1 \
  --enable-priority-scheduling --schedule-low-priority-values-first \
  --default-priority-value 50 --priority-scheduling-preemption-threshold 0 "${priority_args[@]}" \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --stream-interval 1
