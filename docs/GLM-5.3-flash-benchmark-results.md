# GLM-5.3-Flash (EXL3 2.05bpw + DFlash2): benchmark results, 2026-11-02

## TL;DR

- GLM-5.3-Flash now produces correct output on the DGX Spark GB10.
  - Earlier garbage output (`'The付 付付horshorshors'`) came from 3 corrupt (sparse, partly zero-filled) weight shards: `model-00009/10/11-of-00012`.
  - The image, exllamav3 and overlay were fine. The shards were re-downloaded from the pinned revision 51058cd and sha256-verified (15/15 LFS files OK).
- **tool-eval-bench: 87/100 (Good), 113/130 points, 69 scenarios.** Three of them (TC-38/39/40) were rejected because they exceed the 8192 context.
- **llama-benchy pp2048/tg128 at depth 0: PP 418 tok/s, TG 21.2 tok/s, TTFT 5.2 s.** At depth 4096: PP 468 tok/s, TG 13.6 tok/s.
- Compared with Qwen3.8-27B-NVFP4-DFlash2 (91/100): slightly faster decode, about 4.5x slower prefill, much smaller usable context.

## Config used

| Item | Value |
|---|---|
| Image | `glm53-one-spark-vllm:tp1-general23` (local build of gitcommit90/glm-5.3-one-spark @ddc28b1, original, no PR patches) |
| Weights | turboderp/GLM-5.3-Flash-exl3, branch 2.05bpw, rev 51058cd (EXL3 2.05 bpw, mul1 codebook) |
| Drafter | incoai/GLM-5.3-Flash-DFlash2 rev bf582e4, K=5 |
| Serve script | recipe `scripts/serve-one-spark.sh` from clone branch `local/prs-4-5`, which applies the PR #4 mamba prefix-hit seed fix at container start |
| KV | fp8, prefix caching on |
| Speed legs | GMU 0.80, ctx 8192, max-num-seqs 4: KV 11.55 GiB = 11,673 tokens |
| Tool-eval leg | GMU 0.78, ctx 8192, max-num-seqs 1: KV 11.65 GiB = 11,776 tokens (now the launch-script default) |
| Memory guard | floor 3 GiB (`GUARD_FLOOR_GIB`) |
| EXL3 MoE tier | configured `kernel`, effective `sorted` (`shared_suh_absent`). The checkpoint has no shared-SUH layout, so this fallback is expected (recipe README) and affects speed only, not correctness. |

The higher ladder rungs could not start. Even after drop_caches, vLLM saw only 99.4–99.9 GiB free at startup, which is below 0.82 x 121.69 = 99.78 GiB and 0.85 x 121.69 = 103.43 GiB. vLLM's view is about 5–6 GiB lower than a bare torch `mem_get_info` probe.

## Correctness check (temp 0)

| Prompt | Thinking off | Thinking on |
|---|---|---|
| Capital of France | `Paris` | `Paris` (reasoning OK) |
| Count 1–20 | `1, 2, 3, … 20` | same |
| `is_prime` code | correct function | correct function in a fenced block |
| Tool call (weather in Berlin) | `get_weather {"city":"Berlin"}` | same |
| Bat-and-ball puzzle | `The ball costs **$0.05**.` | `**5 cents.**` |

DFlash2 acceptance:
- On the check prompts: mean acceptance length 4.1–5.6 of 6 (62–93%).
- Over the whole tool-eval run: 3.93 (58.5%).
- Before the weight repair: 1.0–1.7.

## tool-eval-bench (3.0.1.dev9 local checkout, 69 scenarios, temp 0, seed 42, `--no-think`, timeout 120 s)

Score 87/100, ★★★★ Good. 113/130 points, completion rate 94.2%, deployability 69, median turn 5.7 s.
DB: `data/benchmarks.sqlite` run `2026-11-02T20-15-29.608034Z_cf3ba41f`.
Report: `test-results/quality/2026/11/2026-11-02T20-15-29.608034Z_cf3ba41f--glm-5.3-flash-exl3-2.05bpw-dflash2-k-5-image-tp1-general23-ctx-8192.md`.

| Category | % |
|---|---:|
| Tool Selection | 100 |
| Parameter Precision | 100 |
| Multi-Step Chains | 100 |
| Restraint & Refusal | 83 |
| Error Recovery | 100 |
| Localization | 100 |
| Structured Reasoning | 100 |
| Instruction Following | 100 |
| Context & State | 60 |
| Code Patterns | 100 |
| Safety & Boundaries | 85 |
| Autonomous Planning | 67 |
| Creative Composition | 83 |
| Structured Output | 92 |
| Toolset Scale | not scored (TC-38/39/40 exceed the 8192 context, HTTP 400) |

## llama-benchy (0.4.1.dev1, pp2048, tg128, concurrency 1, 2 passes x 3 runs, thinking off)

| Depth | PP tok/s | TG tok/s | TTFT s |
|---:|---:|---:|---:|
| 0 | 418 | 21.2 | 5.22 |
| 4096 | 468 | 13.6 | 13.54 |

Depth 16384, the standard depth for other models, cannot run at ctx 8192. Raw files: `test-results/benchmarks/GLM-5.3-Flash-EXL3-2.05_20261102_204055_pass1.json` and `_204252_pass2.json`.

## Comparison with all previously benchmarked models

Tool-eval rows are each model's latest completed run in `data/benchmarks.sqlite`. Speed rows are from the latest llama-benchy JSON in `test-results/benchmarks/` (pp2048/tg128, c=1).

Earlier runs used different tool versions and scenario counts (15 = `--short`, 69 = full). Only rows with the same scenario count are directly comparable.

### Tool use

| Model | Date | Scenarios | Tool version | Score | Points | Deployability | Median turn s |
|---|---|---:|---|---:|---|---:|---:|
| **GLM-5.3-Flash-EXL3-2.05** | 2026-11-02 | 69 | 3.0.1.dev9 | **87** | 113/130 | 69 | 5.7 |
| Qwen3.8-27B-NVFP4-DFlash2 | 2026-09-22 | 69 | 2.7.0 | 91 | 126/138 | 78 | 3.3 |
| Qwen3.8-27B-Uncensored-NVFP4-DFlash2 | 2026-09-22 | 69 | 2.7.0 | 88 | 122/138 | 76 | 3.1 |
| Qwen3.8-27B-Uncensored-OrcaRouter-NVFP4 | 2026-09-23 | 69 | 2.7.0 | 88 | 122/138 | 67 | 8.4 |
| Qwen3.6-27B-PrismaQuant-5.5bit | 2026-07-07 | 69 | 2.1.0 | 87 | 120/138 | 64 | 14.6 |
| Nemotron-3-Nano-30B-A3B-NVFP4 | 2026-07-07 | 69 | 2.1.0 | 75 | 104/138 | 58 | 7.7 |
| GPT-OSS-120B | 2026-07-07 | 69 | 2.1.0 | 71 | 98/138 | 66 | 2.7 |
| Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP | 2026-07-07 | 69 | 1.7.0 | 70 | 96/138 | 54 | 8.8 |
| Qwen3.8-Flash-Next-NVFP4 | 2026-09-04 | 88 (temp 1.0) | 2.6.1.dev | 86 | 152/176 | 66 | 7.5 |
| Kolibri-1 | 2026-10-04 | 15 | 2.7.0 | 100 | 30/30 | 74 | 9.6 |
| Qwen3.6-35B-A3B-FP8 / Qwen3.5-35B-A3B-FP8 / Qwen3.5-122B-A10B-int4-AutoRound | 2026-07/08 | 15 | 2.1.0 | 97 | 29/30 | 75–80 | 4.0–6.4 |
| Qwen3.6-35B-A3B-PrismaQuant-4.75bit | 2026-08-10 | 15 | 2.1.0 | 93 | 28/30 | 80 | 3.2 |
| Qwen3.5-4B-Q4_K_M | 2026-07-09 | 15 | 2.1.0 | 93 | 28/30 | 90 | 1.1 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 2026-07-09 | 15 | 2.1.0 | 90 | 27/30 | 68 | 9.3 |
| Qwen3-Coder-Next-FP8-Dynamic | 2026-07-09 | 15 | 2.1.0 | 87 | 26/30 | 80 | 2.0 |
| DeepSeek-V4-Flash-IQ2XXS-DS4 | 2026-06-16 | 15 | 2.0.0 | 20 | 6/30 | 14 | 56.0 |

Other 15-scenario rows are in the DB.

### Speed (llama-benchy pp2048/tg128, c=1; latest file per model)

| Model | PP d0 | TG d0 | PP d16384 | TG d16384 |
|---|---:|---:|---:|---:|
| **GLM-5.3-Flash-EXL3-2.05** (2-pass mean) | **418** | **21.2** | n/a (ctx 8192; d4096: 468 / 13.6) | n/a |
| Qwen3.8-27B-NVFP4-DFlash2 | 1929 | 18.8 | 1451 | 15.9 |
| Qwen3.8-27B-Uncensored-NVFP4-DFlash2 | 1787 | 15.2 | 1524 | 15.1 |
| Qwen3.8-27B-Uncensored-OrcaRouter-NVFP4 | 1497 | 8.4 | 1185 | 9.0 |
| GPT-OSS-120B | 4733 | 55.6 | 3302 | 40.4 |
| Nemotron-3-Nano-30B-A3B-NVFP4 | 6882 | 51.3 | 6105 | 50.7 |
| Nemotron-3-Super-120B-A12B-NVFP4 | 1303 | 15.0 | 1472 | 14.9 |
| Qwen3.5-122B-A10B-int4-AutoRound | 1182 | 25.4 | 1394 | 24.1 |
| Qwen3.5-122B-A10B-NVFP4 | 955 | 10.4 | 1453 | 10.5 |
| Qwen3.6-35B-A3B-PrismaQuant-4.75bit | 3049 | 40.8 | 4710 | 39.5 |
| Qwen3.6-35B-A3B-FP8 | 1706 | 20.8 | 2319 | 17.5 |
| Qwen3-Coder-Next-int4-AutoRound | 1587 | 38.7 | 1790 | 26.5 |
| Qwen3-Coder-Next-FP8-Dynamic | 1615 | 34.8 | 3520 | 33.0 |
| Qwen3.5-4B-Q4_K_M | 3433 | 48.0 | 3250 | 42.6 |
| TheDrummer GLM-Steam-106B-A12B Q4_K_M GGUF | 538 | 16.5 | – | – |
| DeepSeek-V4-Flash-IQ2XXS-DS4 | 52 | 11.2 | 84 | 10.6 |

## Crash / relaunch log (all times CET)

| Time | Event | Cause |
|---|---|---|
| 15:53–15:59 | Garbage output | Corrupt shards 09/10/11 (found later) |
| 17:05, 17:08 | GLM request queued behind / evicted by Qwen3.8 | Periodic Qwen caller; Qwen unloaded via API |
| 17:08, 17:11 | GLM failed to start at GMU 0.85 | vLLM saw 99.41 / 99.86 GiB free < 103.43 |
| 17:22–17:26 | No-spec diagnostic run exited after 1 request | Most likely the 3 GiB guard (cause not logged) |
| ~20:30 | CUDA-free 14 GiB | 95 GiB page cache from sha256 verification and download; user ran drop_caches |
| 20:35 | GMU 0.82 failed to start | vLLM saw 99.7 GiB < 99.78 |
| 21:00:08 | GLM (0.80 / 8192 / seqs 4) died after tool-eval scenario 50 | MemAvailable was about 3 GiB (integer reading) right before death, which points to a guard EMERGENCY STOP. Not proven: no guard line was captured, and the log was overwritten by the restart. The partial DB row `2026-11-02T19-44-39.774915Z_cceeb1ac` stays `running` and is invalid. |
| 21:11–21:36 | Relaunch at 0.78 / 8192 / seqs 1; full tool-eval completed | MemAvailable min about 3 GiB, no crash |

That makes 1 crash during the benchmark legs, plus 1 relaunch. No Qwen eviction happened during the benchmark legs.

## Caveats / unverified

- GLM is a 2-bit quant; see the recipe README for quantization fidelity. The quality loss is not measured here.
- Context is only 8192. Toolset Scale was not scored and the depth-16384 speed point is missing.
- GLM used tool-eval-bench 3.0.1.dev9 (local checkout); the Qwen3.8 rows used 2.7.0. Scores are not strictly matched (max 130 vs 138 points).
- The speed legs (0.80 / seqs 4) and the tool-eval leg (0.78 / seqs 1) used different configs.
- Not tested: thinking-on tool-eval, contexts above 8192, rungs 0.82 and 0.85 (they need about 100–104 GiB free as seen by vLLM).
- Recipe PRs #4 and #5 were reviewed and built as `glm53-one-spark-vllm:tp1-general23-prs` (clone branch `local/prs-4-5`). The PR #5 image was not needed. The PR #4 serve script is still in use for its prefix-hit seed fix.

## User follow-ups

- Run `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before each GLM launch. Page cache is not counted as free CUDA memory on GB10.
- Restore earlyoom and restart the TTS/STT services if they were stopped for this run.
- Optional: remove `/home/sparky/LLMs/glm53/shard-corrupt/` (24 GB of corrupt shards), the empty `/home/sparky/LLMs/glm53/shard-fix/`, the `:tp1-general23-prs` image, and the `local/prs-4-5` branch. If you remove the branch, the launch script falls back to its sed-based settings on the ddc28b1 serve script.
