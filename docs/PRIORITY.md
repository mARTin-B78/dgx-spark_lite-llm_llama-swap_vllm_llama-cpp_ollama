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
| P4 | P4-Background | 20 | 80 | 1 |

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
their previous behavior. This requires `dgx-spark-local` to route to the FP8
backend; the previous NVFP4 launcher does not enable priority scheduling.

The FP8 launcher schedules lower values first, defaults unspecified priorities
to 50, and disables running-request preemption by default. It orders the queue
but lets an already-running background request finish. Set
`QWEN38_PRIORITY_PREEMPTION=1` in its launch environment (for example,
`cmd: env QWEN38_PRIORITY_PREEMPTION=1 bash /app/scripts/launch-qwen38-huihui-fp8.sh ${PORT} ${host}`) to experiment with
retracting and re-prefilling lower-priority requests; test streamed continuity
and eventual background completion before enabling it for normal traffic.
This scheduler covers one resident model, not independent OCR/STT/TTS processes.

Redeploy: `docker compose up -d litellm`. Rollback: remove the `priority_hook` callback line and volume line
(`LiteLLM/config.yaml.bak2` holds the previous config), then redeploy.
