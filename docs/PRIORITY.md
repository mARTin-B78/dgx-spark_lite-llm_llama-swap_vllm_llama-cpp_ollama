# Request priority tiers

Virtual keys carry `metadata.priority` (higher = more important). `LiteLLM/priority_hook.py`
(pre-call hook) sets the request-body `priority` = `100 - metadata.priority` (clamped 0..100),
overwriting any client value; keys without an integer `metadata.priority` are left untouched.
LiteLLM 1.100.0's router scheduler serves LOWER body priority first.

| Tier | Key alias | metadata.priority | Body priority | max_parallel |
|---|---|---|---|---|
| P0 | P0-Realtime | 100 | 0 | unlimited |
| P1 | P1-Interactive | 80 | 20 | 8 |
| P2 | P2-Normal | 60 | 40 | 4 |
| P3 | P3-Automation | 40 | 60 | 2 |
| P4 | P4-Background | 20 | 80 | 2 |

Check: `python3 tests/test_priority_hook.py`. Configure virtual-key metadata through
the existing LiteLLM key-management interface.
Enable the hook by appending `priority_hook.proxy_handler_instance` to
`litellm_settings.callbacks` in the private `LiteLLM/config.yaml`; preserve
the other callbacks. Compose mounts the hook read-only into `/app`.
Caveat: the scheduler only reorders requests while a model has NO healthy deployment (all cooled
down); with a healthy deployment the request passes straight through (it is not a concurrency limiter).

## Shared local admission and P0 reservation

`gpu_priority_hook` provides a separate Redis priority/FIFO queue before backend dispatch.
With `GPU_PRIORITY_ENABLED=true`, inference calls to `GPU_PRIORITY_ENDPOINTS` share
one active request across keys and workers. The lease lasts through streaming completion.
This covers configured llama.cpp, SGLang and media endpoints regardless of backend
priority support; direct calls, unlisted endpoints and unsupported call types bypass it.
Remote inference does not consume this local slot. Merely routing through LiteLLM
is insufficient: the resolved endpoint must be listed and its call type supported.

One of the 128 waiting positions is reserved for P0 (body priority zero). P1–P4
cannot fill that final position; P0 can still be rejected when all 128 are occupied.
This is shared queue capacity, not a reserved GPU execution slot or latency guarantee.
`GPU_PRIORITY_P0_RESERVED_WAITERS` in the LiteLLM container environment sets the
reservation (default `1`, `0` disables, maximum `128`). With the default Compose
setup, explicitly add this variable to the service environment to override it.
Queue overflow returns 429; queue timeout/storage failure returns 503. Per-key
LiteLLM limits can reject requests before they reach this queue.

The shared gate does not preempt a running request. Because only one request reaches
the backend at a time, enabling SGLang preemption alone cannot let a queued P0
interrupt P4. Keep background generations and OCR/audio batches short; split jobs
at the caller into independently submitted chunks. Keep the resident model stable.
True immediate interruption requires a separate, tested scheduling design or
separate compute; do not disable the shared gate merely to enable backend preemption.
Sustained P0 traffic can starve lower tiers; priority is not fairness.

Use a separate virtual key per application and required tier, for example
`voice-P0`, `chat-P1`, `indexer-P4`. An application needs five keys only if it submits
work at all five tiers. Assign the metadata above server-side; per-key caps multiply
when more keys are created, while the shared queue/reservation stays global.
Do not give a background-only application a P0 key. Store each new credential in its
corresponding KeePass service entry; never print keys or place them in command lines.

For `Huihui-Qwen3.8-27B-abliterated-FP8` and `dgx-spark-local`, the hook additionally
sets `extra_body.priority` from the same validated key metadata after deployment
resolution, which also covers the public `default` alias. LiteLLM flattens
that field into the backend request, so the FP8 SGLang launcher can order its
waiting requests even when the deployment is healthy. Other aliases retain
their previous behavior. Forwarding requires the resolved API URL's hostname
to be exactly `llama-swap`; remote URLs containing that name are excluded.
This requires `dgx-spark-local` to route to the FP8
backend; the previous NVFP4 launcher does not enable priority scheduling.

The FP8 launcher schedules lower values first, defaults unspecified priorities
to 50, and disables running-request preemption by default. It orders the queue
but lets an already-running background request finish. Set
`QWEN38_PRIORITY_PREEMPTION=1` in its launch environment (for example,
`cmd: env QWEN38_PRIORITY_PREEMPTION=1 bash /app/scripts/launch-qwen38-huihui-fp8.sh ${PORT} ${host}`) to experiment with
retracting and re-prefilling lower-priority requests; test streamed continuity
and eventual background completion before enabling it for normal traffic.
This scheduler covers one resident model, not independent OCR/STT/TTS processes.

The [suggested vLLM recipe](https://gist.github.com/jetnet/f8a069c90e40b4fad28ba65c66c91a40)
uses `--scheduling-policy priority` and negative/zero/positive body priorities.
The active local FP8 launcher uses SGLang, so retain its existing flags and the
five key tiers above. `max_parallel_requests` is a per-key admission cap: excess
requests receive HTTP 429 before joining the shared priority queue. The per-key cap
itself reserves no realtime capacity, and multiple keys each get their own cap. The launcher's
`--max-running-requests 1` allows one running request; realtime traffic must wait
for it to finish with the default preemption setting. Retry rejected background
requests with bounded backoff in the caller.

## Deterministic validation

Run `scripts/validate-priority.sh`. It needs Docker and the already cached
`redis:7-alpine` and `ghcr.io/martin-b78/litellm-spark:latest` images. It creates
an isolated disposable Redis instance, publishes no ports, uses no API keys or
GPU, and removes its Redis container when finished. No shared services restart.

The tests hold an active request, submit P4 through P0, and confirm all five
are queued in Redis before releasing the holder. They assert admission order
P0, P1, P2, P3, P4, same-priority FIFO, globally reserved P0 waiting capacity,
overflow rejection and reservation-disabled behavior. An integration check
uses the real key-metadata hook and streaming middleware with simulated backend
responses, checking that only one stream is active and client body priority
cannot override the tier. Failures exit nonzero; a successful run prints `PASS`.

This validates the repository's scheduling implementation. It does not validate
the deployed virtual-key database, authentication, actual GPU latency or backend
preemption. Use the live probe below for observed end-to-end performance.

## Live performance probe

For a repeatable check, use existing P0–P4 keys with `scripts/test-priority.sh`
(`MODEL=dgx-spark-local`, `LITELLM_URL=http://localhost:4000`, and `P0_KEY` through
`P4_KEY` supplied through environment variables). This uses the shared GPU when run
against the real service. Timing observations alone do not prove backend queue ordering.
Run `MODEL=dgx-spark-local scripts/test-priority.sh` after securely loading the five
keys into the environment. Five distinct keys are required; no keys are created.
The probe first authenticates each key against `/key/info` and verifies metadata
priorities 100, 80, 60, 40, 20. It reports only tier, numeric priority/cap and
validation status, never the raw response (which can contain the key itself).
Failed preflights include HTTP `status` and a fixed `error` code. For example,
`401`/`http_error` means authentication was rejected, while `200`/`missing_priority`
means key information was readable but lacked the required priority metadata.
`0`/`transport_error` means no usable HTTP response was received. These diagnostics
do not print server response bodies or exception details.
Tier keys are normally `llm_api_routes`-only and receive 403 on `/key/info`. In that
case also export `LITELLM_MASTER_KEY`: the probe then reads key metadata and
`/model/info` with it (via `POST /v2/key/info`, key in the body) and still sends all
inference with the tier keys. The master key is never printed or placed in a URL.
P4 must allow at least two parallel requests, or be unlimited, for the blocker
and its probe. Invalid or unreadable key configuration stops before inference.

`/model/info` must expose the requested model with all matching deployment hosts
equal to `--expected-backend-host` (default `llama-swap`). Use a concrete local
model alias; unavailable or mismatched routing stops before inference. This
checks the configured route rather than assuming every LiteLLM model uses the GPU.
Every streamed response must also identify that host through LiteLLM's
`x-litellm-model-api-base` response header; missing or different backend evidence
makes the measurement inconclusive. Raw header values are not printed.

After a successful warmup, `--rounds 3` measures each tier alone, then submits
P4 through P0 while a longer P4 stream is active, repeated three times. JSON lines
report HTTP status, TTFT, elapsed time, stream gaps/termination and per-tier
baseline/loaded p50/p95 summaries. Unique prompts avoid exact response-cache hits.
TTFT starts at the request and ends at the first content or reasoning chunk;
stream gaps measure time between such chunks, rather than per-token GPU timing.
Overlap is checked when each probe starts; missing overlap makes the run inconclusive.
These are client-observed times through the local backend, not CUDA kernel timings.

For an explicit latency target, run
`MODEL=dgx-spark-local scripts/test-priority.sh --rounds 3 --p0-max-ttft 5`.
The five-second limit is an example: choose the maximum acceptable delay for your
application. Without a limit, the probe measures latency but does not certify a
latency target. Observed first-token order is diagnostic, not proof of queue order.
Exit `1` means configuration/request/stream failure or a failed latency target;
exit `2` means inconclusive overlap/retries; exit `0` means the enabled checks passed.

Use `--help` for bounded token, timeout and retry settings. Slow HTTP headers can
exceed an individual request timeout; a process watchdog bounds the complete run.
Keep model, keys, settings and other GPU traffic consistent between runs. Three
rounds are a smoke test, not a reliable tail-latency estimate; use up to ten rounds
for more samples. Audio/OCR APIs and semantic continuity after preemption are not
covered. Changing backend preemption requires a separately approved service change
and is not performed by the script.
Do not run the key-creation helper merely to test: it creates credentials and
prints them, and credential handling must follow the KeePass policy.

Redis backs shared LiteLLM coordination counters through
`general_settings.coordination_redis`; it does not enable response caching.
Add this mapping under the existing `general_settings` in `LiteLLM/config.yaml`:

```yaml
general_settings:
  coordination_redis:
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
```

The isolated `redis:7-alpine` service publishes no ports, limits the container
to 256 MiB, and caps Redis data at 128 MiB.
The Redis service keeps these counters ephemeral, so restarting it resets
in-flight admission accounting. Its `noeviction` policy preserves existing
counters when memory is full, but new writes fail; monitor memory and errors.
Apply the updated Compose service and private LiteLLM configuration by
redeploying the local Portainer stack before relying on shared counters.

P0 reservation activation requires an approved LiteLLM restart after all workers
have the updated files. Mixed old/new workers cannot guarantee the reserved space.
Rollback reservation alone: set `GPU_PRIORITY_P0_RESERVED_WAITERS=0` in the service
environment and recreate LiteLLM after approval; preserve the existing queue/hooks.

Redeploy: `docker compose up -d litellm`. Rollback: remove the `priority_hook` callback line and volume line
(`LiteLLM/config.yaml.bak2` holds the previous config), then redeploy.
