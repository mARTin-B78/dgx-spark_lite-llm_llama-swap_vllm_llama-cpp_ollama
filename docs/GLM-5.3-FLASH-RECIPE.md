# GLM-5.3-Flash EXL3 2.05bpw + DFlash2 on one DGX Spark (GB10)

Status: F = verified on this host (2026-11-02/03), U = unverified. Details and raw numbers:
`docs/GLM-5.3-flash-benchmark-results-v2.md`.

## 1. Sources (F)
- Image/serve scripts: `gitcommit90/glm-5.3-one-spark` @ `ddc28b1`, plus the serve-script behaviour of its PR #4 (`ONE_SPARK_*` env knobs). Checkout: `/home/sparky/Docker/glm-5.3-one-spark`.
- ExLlamaV3: commit `e648f1a` (pinned by the image build).
- Weights: `turboderp/GLM-5.3-Flash-exl3`, branch `2.05bpw`, rev `51058cd551c7e570d87bd32a4adee720edce2349`.
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, rev `bf582e4eacc1810f76656d1811693ff6c6737d2a`. License CC BY-NC-ND 4.0: non-commercial use only, no derivatives. Check before any commercial use.

## 2. Disk and download (F)
About 85 GB under `/home/sparky/LLMs/glm53/` (`GLM-5.3-Flash-exl3-2.05bpw`, `GLM-5.3-Flash-DFlash2`).
```
export HF_HUB_DISABLE_XET=1          # xet produced sparse/corrupt shards here
hf download turboderp/GLM-5.3-Flash-exl3 --revision 51058cd551c7e570d87bd32a4adee720edce2349 \
  --local-dir /home/sparky/LLMs/glm53/GLM-5.3-Flash-exl3-2.05bpw
hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a \
  --local-dir /home/sparky/LLMs/glm53/GLM-5.3-Flash-DFlash2
```
Verify all 12 weight shards: `sha256sum *.safetensors` against the HF LFS oids (model page, "Files", or the API `lfs.oid`). 3 shards were sparse (correct size, zeros) and gave garbage output. Always verify.

## 3. Build the image (F)
No prebuilt image exists.
```
cd /home/sparky/Docker/glm-5.3-one-spark && git checkout ddc28b1
docker build -t glm53-one-spark-vllm:tp1-general23 .
```

## 4. Settings that gave 93/100 (F)
GMU 0.76-0.78 (script default 0.76), ctx 262144, seqs 1, `ONE_SPARK_ASYNC=0`, `ONE_SPARK_DRAFT_BLOCK=1024`, `GLM53_INDEXER_WORKSPACE=rightsize`, `ONE_SPARK_K=5`, `ONE_SPARK_MAMBA_SEED_FIX=1`. All are defaults in `llama-swap/scripts/launch-glm53-flash.sh`; overrides via untracked `llama-swap/runtime/glm53.env` (`GLM53_GMU`, `GLM53_CTX`, `GLM53_SEQS`, `GLM53_IMAGE`).

## 5. Memory ladder (F)
vLLM needs GMU x 121.7 GiB free in cudaMemGetInfo at start. Page cache is NOT counted as free by CUDA on GB10: run `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before launch (needs about 103-105 GiB free after unloading other models).
- 0.90 (recipe default): starves the host.
- 0.80 and 0.78 at ctx 262144, and 0.76 with seqs 4: stopped by the memory guard under load.
- 0.76 / seqs 1 / async off / draft block 1024 / rightsize: stable, min MemAvailable 3.29 GiB, KV 5.56 GiB (376k tokens).
- Blockers: page cache, guard floor (default 10 GiB; GLM opts in to 3 via `GUARD_FLOOR_GIB`, see `guard-large-model.sh`), earlyoom `-m 8,4`, desktop apps. TTS/STT must be stopped during GLM runs.

## 6. llama-swap entry (F, `llama-swap/config.yaml`)
```yaml
  GLM-5.3-Flash-EXL3-2.05:
    ttl: 0
    readyTimeout: 1800
    checkEndpoint: "/health"
    cmd: bash /app/scripts/launch-glm53-flash.sh ${PORT} ${host}
    cmdStop: "docker stop -t 30 vllm-glm53-flash-${PORT}"
```
LiteLLM (`LiteLLM/config.yaml`; key comes from the environment, never inline):
```yaml
- model_name: GLM-5.3-Flash-EXL3-2.05
  litellm_params:
    model: openai/GLM-5.3-Flash-EXL3-2.05
    api_base: http://llama-swap:8080/v1
    api_key: os.environ/LITELLM_MASTER_KEY
    max_tokens: 2048
    drop_params: true
  model_info: {max_input_tokens: 120000, max_output_tokens: 4096}
```

## 7. Benchmarks (F)
Checkout `tool-eval-bench` 3.0.1.dev9 (`Docker/tool-eval-bench`, its own `.venv`; the PATH version 2.7.0 has only 88 scenarios).
```
tool-eval-bench --hardmode --seed 42 --parallel 1 --context-size 262144 --no-warmup --no-live   # thinking default
tool-eval-bench bench --perf-only      # llama-benchy: pp2048 tg128, depths 0/4096/8192, 3 runs
```
| Test | Result |
|---|---|
| tool-eval-bench, 104 scenarios, seed 42 | 93/100 (combined 192/208) |
| llama-benchy c=1 decode | 28.1 / 27.8 / 26.8 tok/s (depth 0 / 4096 / 8192) |
| llama-benchy c=1 prefill | 499 / 549 / 565 tok/s |
| c=2, c=4 | decode 19.1 and 22.5 at depth 0; c=4 run hit the guard, treat as unreliable |
Forum reference: 29.7 tok/s decode, 571 prefill at c=1.

## 8. Troubleshooting
- Garbage output: sha256 of the shards (F, cause of the 2026-11-02 failure).
- Refuses to start (not enough free memory): drop_caches, unload other models, stop TTS/STT (F).
- Container killed mid-run: check MemAvailable vs the guard floor, see section 5 (F). Docker does not report OOMKilled; exit 137 is in the events log.
- Qwen reloads at :05/:20/:35/:50 and evicts GLM: a LiteLLM client issues periodic Qwen requests (observed pattern; exact client not identified, U). Stop that client during GLM runs.

## 9. Caveats
- Memory-limited: seqs 1 only is stable; c>1 numbers are unreliable and llama-benchy's c>1 aggregation is unverified (U).
- 4 runs in `data/benchmarks.sqlite` carry a wrong `server_model_id`; GLM served all requests (llama-swap logs). Cause unproven (U).
- Drafter license is non-commercial.
