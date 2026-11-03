# GLM-5.3-Flash v2: reproduction of the forum 93/100 (2026-11-03)

Forum reference: sson, https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140/19 (image only: tool-eval-bench v3.0.1.dev7, 104 scenarios incl. Hard Mode, seed 42, 256K ctx, 93/100).

## TL;DR

- **Reached 93/100 (Excellent), 193/208 points, all 104 scenarios, no infrastructure exclusions.** Fact: Run `2026-11-03T08-38-04.500300Z_becf9b1d`.
- The difference to our earlier 87 is mostly **thinking mode** (inference, supported by a controlled run): the same setup with `--no-think` scores **82/100** (run `..._31864cef`). The old 87 was a no-think run on only 69 scenarios.
- The run was assembled from 3 segments (2 crashes, resumed from checkpoints, see Crash log). The first 52 scenarios ran at GMU 0.78, the remaining 52 at GMU 0.76. Sampling settings were identical.
- Speed is below the forum: decode 24-27 tok/s (forum 29.7), prefill ~460 tok/s (forum 571). c=4 aggregate not reproduced (see Speed).
- GMU 0.80 (the planned value) is not stable on this host: the 3 GiB MemAvailable guard stops it. Launch-script defaults now: GMU 0.76, ctx 262144, seqs 1.

## Config (final, stable)

| Item | Value |
|---|---|
| Image | `glm53-one-spark-vllm:tp1-general23`, serve script from clone branch `local/prs-4-5` (PR #4 mamba seed fix) |
| GMU / ctx / seqs | 0.76 / 262144 / 1 (also 0.78 for the first 52 scenarios; 0.80 and seqs 4 were tried, see below) |
| ONE_SPARK_ASYNC | 0 (`--no-async-scheduling`) |
| ONE_SPARK_DRAFT_BLOCK | 1024 (log: `drafter block patch: 1 + 1 sites`, confirmed applied) |
| GLM53_INDEXER_WORKSPACE | rightsize (log: 262152 entries instead of 10485760, ~1287 MiB reclaimed, confirmed) |
| ONE_SPARK_K | 5, ONE_SPARK_MAMBA_SEED_FIX=1 |
| KV | fp8; GMU 0.80: 14.9 GiB / 1,014,083 tokens (3.87x at 262144; seqs 1 first boot) and 1,069,271 tokens; GMU 0.78: 11.5 GiB / 782,982 tokens (2.99x); GMU 0.76: **5.56 GiB / 375,969 tokens (1.43x)**; GMU 0.76 seqs 4: 11.26 GiB / 765,736 tokens (2.92x) |
| Host free | MemAvailable after init: ~4.6 GiB (0.80), 3.4-9 GiB (0.78), 6-12 GiB (0.76). Min during stable run (0.76, seqs 1, ~80 min): **3.29 GiB** |
| CUDA free before launch | 104.9 GiB by torch (other models unloaded) |
| tool-eval-bench | `3.0.1.dev9+g2ad584631` (checkout `v3.0.0-9-g2ad5846`, dirty), the forum used dev7 |
| Flags | `run --hardmode --seed 42 --parallel 1 --context-size 262144 --no-warmup --no-live`; temperature 0.0, timeout 120 s, max turns 8 |
| Thinking | default = **enabled**, max tokens 16384 (CLI defaults). The 82 run used `--no-think`: thinking disabled, max tokens 4096 |

Env overrides were applied via `llama-swap/runtime/glm53.env` (not committed, kept as `glm53.env.v2-forum-repro-used`).

## Scores per category

| Category | Forum | Ours v2 (think, 93) | Ours v2 no-think (82) | Ours v1 (87, 69 sc., no-think) |
|---|---|---|---|---|
| A-H (selection, params, chains, restraint, recovery, locale, reasoning, instr.) | 100 | 100 | 100 / 100 / 100 / 83 / 100 / 100 / 100 / 80 | 100 (restraint 83) |
| I Context & State | 95 | 90 | 65 | 60 |
| J Code Patterns | 100 | 100 | 100 | 100 |
| K Safety & Boundaries | 92 | 92 | 85 | 85 |
| L Toolset Scale | 88 | 88 | 88 | (rejected, ctx 8192) |
| M Autonomous Planning | 67 | 50 | 67 | 67 |
| N Creative Composition | 100 | 100 | 100 | n/a |
| O Structured Output | 92 | 83 | 92 | n/a |
| P Hard Mode | 89 | 93 (65/70) | not split out | n/a |
| **Total** | **93** | **93** | **82** | **87** |

Our v2 non-perfect scenarios: TC-40, 43, 51, 52, 62, 63, 67, 68, 92, 99, 104. Per-category differences to the forum are within one or two scenarios (single-seed noise cannot be excluded).

## Speed (llama-benchy 0.4.1 via `tool-eval-bench bench --perf-only`, pp2048 tg128, 3 runs)

| Config | depth | c | pp tok/s | tg tok/s |
|---|---|---|---|---|
| 0.76, seqs 1 | 0 | 1 | 466 | 24.4 |
| 0.76, seqs 1 | 4096 | 1 | 454 | 18.0 |
| 0.76, seqs 1 | 8192 | 1 | 459 | 22.0 |
| 0.76, seqs 4 | 0 | 1 / 2 / 4 | 457 / 266 / 940 | 26.9 / 16.9 / 17.9 |
| forum | - | 1 / 4 | 571 | 29.7 / 53.5 aggregate |

The seqs 4 point is **unreliable and not a reproduction of the forum's 53.5**: the container was stopped by the memory guard at depth 8192 (see Crash log) and how llama-benchy aggregates tg at c>1 was not verified. The forum's prefill/decode gap (about 20 percent) may come from async scheduling being off (set to cut KV need), a different GMU, or host load. Not tested.

## Crash log

All stops were MemAvailable falling to the 3 GiB guard floor (`GUARD_FLOOR_GIB=3`) or, in the first case, an idle death whose cause is unknown. Evidence: `/home/sparky/LLMs/glm53/overnight/` (rolling vLLM log copies, `v2-memavail*.txt`, `v2-docker-events.log`). The runtime log is overwritten by each relaunch, so the first stops are inferred from MemAvailable samples (2.85 GiB at the stop) rather than a recorded `[memory-guard] EMERGENCY STOP` line.

| # | Config | Event | Evidence |
|---|---|---|---|
| 1 | 0.80 / 262144 / 1 | Idle after load: MemAvailable drifted 4.6 to 3.2 GiB over 17 min, container died (llama-swap: "exited unexpectedly"), cause not recorded | memavail samples |
| 2 | 0.80 / 262144 / 1 | Died 15 s into the first scenario, MemAvailable 2.85 GiB | memavail samples |
| 3 | 0.78 / 262144 / 1 | Died around scenario 56 (min sampled 3.57 GiB at 15 s resolution, a short dip is likely) | rolling log |
| 4 | 0.78 / 262144 / 1 | Died during TC-77 with a 370 tok/s prefill request, MemAvailable 3.00 GiB, docker `kill` then exit 137 | v2-docker-events.log (1 s sampling) |
| 5 | 0.76 / 262144 / **4** | Died at depth 8192 during perf, MemAvailable 2.96 GiB, `kill` + exit 137 | v2-docker-events.log |

0.76 with seqs 1 ran the final 30 scenarios (resume), the whole 104-scenario no-think run (38 min) and the perf run without a stop. OOMKilled was not reported by Docker for any stop (the container is `--rm`, so `docker inspect` after the fact fails; the exit code 137 comes from the events log). The scores of the interrupted run were restored with `tool-eval-bench resume` (checkpointed outcomes kept, infra failures re-run).

Note: relaunches exceeded the planned 3 because the cause was a config issue (host memory), not flakiness; GMU was lowered stepwise.

## Facts versus inference

- Fact: 93/100 reached on all 104 scenarios, with the forum's flags and seed; the key env knobs were verified in the log.
- Fact: thinking on 93 vs off 82 with otherwise identical settings.
- Inference: the old 87 vs new 93 is mostly the thinking mode and the larger max-tokens (16384 vs 4096); also 35 more scenarios and the ctx limit are not comparable.
- Inference: the forum's M 67 and O 92 versus our 50 and 83 are within seed/version noise (dev7 vs dev9).
- Unverified: the forum's exact thinking setting and max tokens, whether they ran at GMU 0.80 stably (they probably had more free host memory), c=4 aggregate speed, the first-crash cause.
- Unverified: a single pass was run per mode; no repeat for variance.

## Defaults and follow-ups

The launch script defaults were changed to the stable config (0.76 / 262144 / seqs 1 / async 0 / draft block 1024 / rightsize). The host baseline (~18 GiB used by other services plus the 3 GiB guard) leaves no room for 0.80. To go higher, free host memory first (drop page cache, stop idle services) or test seqs 4 at a lower GMU. TTS/STT stay stopped and earlyoom stays off as the user set them.

## v3 (uninterrupted run)

Run `2026-11-03T11-46-11.863574Z_56b6369f`, GLM-5.3-Flash-EXL3-2.05, GMU 0.78, ctx 262144, seqs 1, `ONE_SPARK_ASYNC=0`, draft block 1024, K=5, indexer rightsize, thinking enabled (CLI default). Flags: `run --hardmode --seed 42 --parallel 1 --context-size 262144 --no-warmup --no-live`. Result line: `Score: 95 / 100 - Excellent`, 167/176 points, 88/88 scenarios, 3062 s, no infra failures, no container kill during the run (min MemAvailable 4.89 GiB).

### Score per category (v3 vs v2 vs forum)

| Cat | v3 (88 scn, 2.7.0) | v2 `becf9b1d` (104 scn, dev9) |
|---|---|---|
| A-F, H, J, N | 100% each | 100% each |
| G Structured Reasoning | 4/6 (67) | 6/6 |
| I Context & State | 19/20 (95) | 18/20 |
| K | 26/26 | 24/26 |
| L | 7/8 | 7/8 |
| M Autonomous Planning | 4/6 (67) | 3/6 (50) |
| O Structured Output | 9/12 (75) | 10/12 |
| P Hard Mode | 38/38 (19 scn) | 65/70 (35 scn) |
| Total | 167/176 = **95** | 193/208 = **93** |

Forum: 93 (M 67, O 92). Non-pass in v3: TC-21 fail, TC-51 fail, TC-68 fail, TC-40/63/67 partial.

### 88 versus 104 scenarios (fact)

The v3 driver called the `tool-eval-bench` on PATH (`~/.local/share/uv/tools`, **v2.7.0**), which ships 88 scenarios with `--hardmode`. The v2 run used the dev9 checkout (`3.0.1.dev9+g2ad584631`, `Docker/tool-eval-bench/.venv`), whose Hard Mode adds **TC-89..TC-104** (16 scenarios, 32 points; modules hardmode_transactional/documents/expanded/governance). So the 95 run simply did not contain those 16; TC-01..TC-88 are identical IDs in both runs. Also recorded: `server_model_id` in the 95 run metadata says `DeepSeek-V4-Flash-0731-IQ2XXS-DS4` (stale id lookup in 2.7.0; requests went to GLM, the model field in config and report is GLM).

### Is 95 comparable to 93? (fact + inference)

- Fact: on the common TC-01..TC-88 the v2 run scored 166/176 versus 167/176 for v3, i.e. identical within one point. Five scenarios flipped each way (TC-21, 43, 52, 62, 68).
- Fact: v2 on TC-89..104 scored 3 non-passes (TC-92 fail, TC-104 fail, TC-99 partial), so a like-for-like 104 score for v3 needs those 16 run.
- Inference: 95 vs 93 is not a real difference; it is a different scenario set plus single-pass temp-0 noise. Not comparable as printed. A run of TC-89..104 on the same config is in progress (`/home/sparky/LLMs/glm53/overnight/v3b/`); combined score (two runs, computed) is not yet available when this was written.

### Speed (llama-benchy 0.4.1.dev1 via `tool-eval-bench bench --perf-only`, pp2048 tg128, 3 runs)

| Config | depth | c | pp tok/s | tg tok/s |
|---|---|---|---|---|
| 0.78, seqs 1 | 0 / 4096 / 8192 | 1 | 499 / 549 / 565 | 28.1 / 27.8 / 26.8 |
| 0.76, seqs 2 | 0 / 4096 / 8192 | 2 | 309 / 427 / 484 | 19.1 / 18.8 / 17.6 |
| 0.76, seqs 4 | 0 / 4096 / 8192 | 4 | 416 / 407 / 449 | 22.5 / 12.8 / 9.3 |
| forum | - | 1 / 4 | 571 | 29.7 / 53.5 aggregate |

c=1 reaches about 95 percent of the forum decode (28.1 vs 29.7) and 87-99 percent of prefill. No guard kill at seqs 2 or 4; min MemAvailable during these passes 6.0 GiB (seqs 4), 7.8 GiB (seqs 2), 10.3 GiB (seqs 1). Not verified whether llama-benchy's c>1 tg is per request or aggregate; the c=4 figures here are far below the forum's 53.5 aggregate if per-request, and would not match it if aggregate either, so no claim is made. These passes ran after the model had been reloaded, with the GPU free of other load.

### Driver incident (fact)

`driver.sh` could not run `dbq.py` (relative path, wrong cwd), so it parsed `done=0 status=none` after the finished run (rc 0) and logged `KILL #1 cause: bench ended with infra failures` at 13:37:22, then killed and relaunched GLM. That was a false positive. `status.txt` of that driver is unreliable (0/104, run id `-`). The driver was stopped by pid. The v3b runner uses the real run id from the DB and checkpoint counts.
