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
