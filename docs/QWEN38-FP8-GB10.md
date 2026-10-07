# Huihui Qwen3.8 FP8 on GB10

Checkpoint: [munekazu/Huihui-Qwen3.8-27B-abliterated-FP8](https://huggingface.co/munekazu/Huihui-Qwen3.8-27B-abliterated-FP8), revision
`badcb1961b646b3bdf1fc9ebdd9727826a55a48f`. This is a community FP8
quantization of Huihui's abliterated weights, with dynamic E4M3 activations and
128×128 weight blocks. Six weight shards total 30,866,866,672 bytes.
The linked Huihui GGUF repository has Q8 integer variants, not FP8 weights.

Download without sending an implicit Hugging Face credential:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download munekazu/Huihui-Qwen3.8-27B-abliterated-FP8 \
  --revision badcb1961b646b3bdf1fc9ebdd9727826a55a48f \
  --local-dir /home/sparky/LLMs/qwen38/munekazu/Huihui-Qwen3.8-27B-abliterated-FP8 \
  --max-workers 2
```

The local ignored configs contain the new model entry; reproduce it in
`llama-swap/config.yaml` under `models`:

```yaml
  Huihui-Qwen3.8-27B-abliterated-FP8:
    aliases: [dgx-spark-local]
    useModelName: Huihui-Qwen3.8-27B-abliterated-FP8
    cmd: bash /app/scripts/launch-qwen38-huihui-fp8.sh ${PORT} ${host}
    cmdStop: docker stop qwen38-huihui-fp8-${PORT}
    checkEndpoint: /health
    readyTimeout: 1200
    ttl: 0
```

Remove the previous standalone `dgx-spark-local` model block and set
`includeAliasesInList: true` at the top level. Both names then share one
resident FP8 model rather than starting duplicate containers.

And in `LiteLLM/config.yaml` under `model_list`:

```yaml
- model_name: Huihui-Qwen3.8-27B-abliterated-FP8
  litellm_params:
    model: openai/Huihui-Qwen3.8-27B-abliterated-FP8
    api_base: http://llama-swap:8080/v1
    api_key: os.environ/LITELLM_MASTER_KEY
    max_tokens: 4096
    drop_params: true
  model_info:
    max_input_tokens: 61440
    max_output_tokens: 4096
```

The existing LiteLLM `dgx-spark-local` entry also uses `max_tokens: 4096` and
the same `model_info` limits, matching its shared 64K backend. With
`STORE_MODEL_IN_DB=true`, the database router alias also needs updating:
`default` now resolves to `Huihui-Qwen3.8-27B-abliterated-FP8`. The previous
alias map is saved in `/tmp/qwen38_default_alias_before_fp8_20261006.json`.

Initial profile: FP8 weights, BF16 KV cache, 64K context, one request, no
speculative draft. BF16 KV avoids using uncalibrated FP8 cache scales. The existing
large-model guard checks available host RAM and stops its own model below
10 GiB available. It does not reserve memory against independent OCR/TTS jobs.

Activation was authorized on 2026-10-06: drain/unload the existing Qwen
backend and restart the shared llama-swap and LiteLLM routers. Check `/health`, then run
text, streaming, and tool-call smoke requests through the new alias before
using `dgx-spark-local`. All six shard hashes passed; runtime logs confirmed
E4M3 FP8 weights and a BF16 KV pool of 65,536 tokens. Arithmetic inference
passed through the canonical model, `dgx-spark-local`, and LiteLLM `default`.
Streaming returned content and `[DONE]`; forced `report_sum(a=2,b=3)` returned
valid arguments. A live priority test completed the running background request,
then the queued foreground request, then the queued background request, as intended.
Rollback: restore the database alias map, remove the two FP8 entries, restore the former
standalone `dgx-spark-local` block (saved locally in
`/tmp/qwen38_default_before_fp8_20261006.yaml`), remove `dgx-spark-local` from
the hook's backend-forwarding names, then restart the routers and request the
previous NVFP4 alias. Its model definition remains available.

The pinned SGLang image has a stream-cancellation cleanup bug: its exception
handler removes request state before the delayed abort, and the abort then
returns because state is absent. This can leave generation running and emit
`state was deleted in TokenizerManager` repeatedly. The launcher applies an
exact-match, compile-checked patch inside the disposable container: dispatch
abort before deleting state. Abort-send failures are logged while cleanup
continues. After deliberately disconnecting a long stream, a follow-up arithmetic
request completed in 0.44 seconds. Two in-flight output warnings occurred, without
the earlier sustained orphan generation. Installed cleanup-helper tests also
verified that abort-send failures do not prevent cleanup of later batch requests.

# Compute priority

The [forum discussion](https://forums.developer.nvidia.com/t/prioritizing-compute-on-the-gb10/385139)
reports that metadata priorities did not solve contention. The previous
LiteLLM hook only reordered requests when deployments were unavailable;
healthy traffic bypassed that queue. The hook now forwards validated key
priorities to SGLang for the two FP8 aliases, and the launcher enables its
backend queue ordering. Running-request preemption is off by default;
`QWEN38_PRIORITY_PREEMPTION=1` opts into a separate continuity experiment.
See `docs/PRIORITY.md` for the scope and limits.

Recommended first experiment: admit at most one background generation at a
time, shorten background batches/prefills, defer bulk OCR while voice/chat is
active, and keep the interactive model resident. This is application admission
control; separate OCR and voice endpoints must participate independently.
Measure interactive time to first token and voice latency with the same
background workload before and after each change; no speedup is claimed yet.

GB10 is absent from NVIDIA's [MIG supported GPU list](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html).
Docker CPU/memory quotas do not partition GPU compute.
[CUDA stream priority](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)
is a scheduling hint and does not preempt an already-running kernel.

[MPS client priority and active thread percentage](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html)
are a possible second experiment, after confirming MPS works on this GB10 and
driver. They require participating clients to start under MPS; setting an
environment variable on existing containers alone is insufficient. These
controls do not guarantee memory-bandwidth isolation. No MPS daemon, service
restart, container pause, or system-wide scheduling change was performed.

# VeloGB10 findings

The [v0.4.0 thread](https://forums.developer.nvidia.com/t/velogb10-v0-4-0-release-qwen-3-8-27b-code-sustained-tok-s-4x-125-tok-s-2x-85-tok-s-1x-75tok-s/381027)
uses TP=1/2/4: one, two, or four GB10 machines accelerating a single request.
Its code-generation figures use NVFP4 plus DFlash2; they are not FP8 Huihui
measurements or general chat throughput guarantees.

Current upstream has moved beyond that release. Its [README](https://github.com/sf-stav/veloGB10)
documents direct block-128 FP8 loading as well as specialized NVFP4 artifacts.
The Huihui checkpoint has the corresponding block layout, making a direct
load experiment plausible, but that does not establish compatibility or
speculative acceptance with modified weights.

Useful next experiment after the SGLang baseline: benchmark the same Huihui
FP8 checkpoint on a pinned VeloGB10 release, through a separate llama-swap
alias, measuring short/long prompt TTFT, decode, tool-call validity, and voice
latency under contention. Test the existing DFlash2 draft separately and
measure acceptance; do not assume the code-demo speed applies to chat.
Prefix caching and speculative decoding are already available in SGLang;
Velo's engine-specific CUDA kernel switches cannot simply be copied into it.
VeloGB10 is an inference engine experiment, not cross-service GPU priority.

## Single-Spark Velo recipe

[v0.7.3](https://github.com/sf-stav/veloGB10/releases/tag/v0.7.3) documents
the [single-node FP8 recipe](https://github.com/sf-stav/veloGB10/blob/v0.7.3/BEST_WAYS_TO_RUN.md).
Run from its matched binary/PTX release directory; no peer nodes or RDMA setup
are needed. Adapted to the local paths and 64K profile:

```sh
./gb10_inference --server \
  --model-dir /home/sparky/LLMs/qwen38/munekazu/Huihui-Qwen3.8-27B-abliterated-FP8 \
  --host 127.0.0.1 --port 5900 \
  --kv-cache k8v8 --max-seq-len 65536 --max-batch 1 --max-tokens 4096 \
  --prefix-cache on --default-presence-penalty 0.0 \
  --mtp auto --fp8-prefill on --reasoning-effort low \
  --df2-block 16 --df2-round-shard off --spec-source dflash2 \
  --draft-dir /home/sparky/LLMs/qwen38/z-lab/Qwen3.8-27B-DFlash2
```

The local BF16 draft's config and tensor shapes match the release's DFlash2
loader, so it is structurally reusable. Huihui's FP8 loader path additionally
requires BF16 `weight_scale_inv` tensors and block-128-compatible matrix shapes;
check tensor headers and runtime behavior, including draft acceptance. This
command is a proposed separate engine comparison, not the deployed SGLang
launcher. Velo was not installed or launched by this change.
