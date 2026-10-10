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
requests receive HTTP 429 rather than joining a priority queue. It reserves no
realtime capacity, and multiple keys each get their own cap. The launcher's
`--max-running-requests 1` allows one running request; realtime traffic must wait
for it to finish with the default preemption setting. Retry rejected background
requests with bounded backoff in the caller.

For a manual check, use existing P0 and P4 keys with `scripts/test-priority.sh`
(`MODEL=dgx-spark-local`, `LITELLM_URL=http://localhost:4000`, and keys supplied
through environment variables). With the P4 cap of one, concurrent P4 requests
may return 429; their elapsed times alone do not prove backend queue ordering.
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

Redeploy: `docker compose up -d litellm`. Rollback: remove the `priority_hook` callback line and volume line
(`LiteLLM/config.yaml.bak2` holds the previous config), then redeploy.
