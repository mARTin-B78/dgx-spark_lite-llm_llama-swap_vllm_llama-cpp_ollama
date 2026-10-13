# Tutorial: prioritizing compute on a shared GB10 (P0–P4 tiers)

How to keep interactive requests (chat, voice) fast while background jobs share the same
GB10, using LiteLLM virtual keys, a small Redis gate and a backend that accepts two concurrent
requests. Everything below was set up and measured on this repository (release 0.16.0).

Related: [docs/PRIORITY.md](PRIORITY.md) (operational reference, tiers, validation and rollback),
`LiteLLM/priority_hook.py`, `LiteLLM/gpu_priority_hook.py`, `LiteLLM/gpu_admission.py`,
`scripts/test-priority.sh`, `scripts/validate-priority.sh`. Background discussion:
https://forums.developer.nvidia.com/t/prioritizing-compute-on-the-gb10/385139

## The short version

### What did not help

- A priority in the LiteLLM virtual key metadata, or in the request body, does not reorder anything while your deployment is healthy. LiteLLM's scheduler only kicks in when *no* deployment is available.
- It is worse than "does nothing": with `coordination_redis` enabled, a `priority` in the body gave me random HTTP 500s under concurrent load in LiteLLM 1.100.0. The scheduler mixes tuples and lists that come back from Redis and then cannot compare them.
- Backend-side priority (vLLM/SGLang) orders the *waiting queue*, but a request that is already running is not kicked out when a more important one arrives. In the vLLM source I run, a running request is only preempted when the KV cache runs out.

### What worked

1. One LiteLLM key per app and priority level (P0 = realtime ... P4 = background).
2. A tiny hook that takes the priority **only from the key** (never from the client) and drops any `priority` the client sends.
3. A small **queue in Redis** in front of the GPU: more important requests go first, and there is **one extra lane that only P0 may use**.
4. Background keys (P4) can only generate **64 tokens per request**, so they never hog the GPU for long.
5. The backend has to be able to run **two requests at once**.

### Result

Qwen3.8-27B NVFP4 with DFlash on SGLang, one GB10, 5 rounds per test:

| Situation | First token for P0 |
|---|---|
| Idle machine | 0.27 s |
| A 256-token background request is running, no P0 lane | about 10.9 s |
| Same, with the P0 lane | **0.35 s** (p95 0.55 s) |

The order was correct (P0, P1, P2, P3, P4) in every round.

### Honest limits

- Nothing is interrupted. P1 to P3 still wait for the running request. Only P0 gets the fast lane.
- Two P0 requests at the same time use both lanes; a third P0 waits.
- Anything that bypasses LiteLLM (direct calls to a container, STT/TTS/OCR on their own ports) is not covered unless you route it through the gate as well.
- This is priority, not fairness: constant P0 traffic can starve the rest.

## How it works

Think of a single-lane road to the GPU with one extra emergency lane. A P4 request can be generating in lane A while a P0 request starts immediately in lane B. It does not interrupt the P4 request, it just does not have to wait for it.

```
apps (one key per priority)
  P0 realtime ----+
  P1 chat --------+
  P2 normal ------+--> LiteLLM --> hook: priority from the key only,
  P3 automation --+                      P4 limited to 64 tokens
  P4 background --+                            |
                                               v
                       +--------------------------------------+
                       | Redis gate                           |
                       |  queue: most important first,        |
                       |         same priority = first come   |
                       |  lane A: anyone, one at a time       |
                       |  lane B: P0 only                     |
                       +------------------+-------------------+
                                          |
                                          v
                       llama-swap --> SGLang / vLLM / llama.cpp
                                      (must allow 2 requests at once)
```

## Tutorial

### 0. What you need

- LiteLLM proxy with Postgres (for virtual keys) and Redis
- llama-swap or any OpenAI-compatible backend that can run two requests at the same time
- a way to mount a Python file into the LiteLLM container (callbacks)

### 1. Create one key per app and level

Use the admin API. A higher number means more important.

```bash
curl -s -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_alias":"voice-P0","metadata":{"priority":100},"max_parallel_requests":null}'
```

I use these five levels:

| Tier | metadata.priority | max_parallel_requests |
|---|---|---|
| P0 realtime | 100 | none |
| P1 interactive | 80 | 8 |
| P2 normal | 60 | 4 |
| P3 automation | 40 | 2 |
| P4 background | 20 | 2 |

`max_parallel_requests` is only a cap (extra requests get a 429). It does not do priority, but it stops one tier from filling the whole queue. Give an app only the levels it really needs, and keep the keys in a password manager.

### 2. The hook

This is the whole idea in one place: the priority comes from the key, the client cannot fake it, and background keys get a token limit.

```python
# priority_hook.py  (registered as a LiteLLM callback)
import os
from litellm.integrations.custom_logger import CustomLogger

class PriorityHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # never forward it: it triggers LiteLLM's buggy scheduler
        data.pop("priority", None)

        cap = int(os.getenv("PRIORITY_BACKGROUND_MAX_TOKENS", "64"))
        meta = getattr(user_api_key_dict, "metadata", None)
        prio = meta.get("priority") if isinstance(meta, dict) else None

        # background keys only
        if cap and "messages" in data and type(prio) is int and prio <= 20:
            for f in ("max_tokens", "max_completion_tokens"):
                if type(data.get(f)) is int and data[f] > cap:
                    data[f] = cap
            if "max_tokens" not in data and "max_completion_tokens" not in data:
                data["max_tokens"] = cap
        return data

proxy_handler_instance = PriorityHook()
```

A shortened answer ends with `finish_reason: "length"`, so your background job knows it should simply continue in the next request.

### 3. The gate (Redis)

The gate is a small piece of Python around a Lua script, so the queue and the two lanes are updated atomically, even with several LiteLLM workers. The heart of it:

```lua
-- a request may only start when it is first in the sorted queue
local first = redis.call('ZRANGE', q, 0, 0)[1]
if member ~= first then return 0 end

-- lane A
local got = redis.call('SET', active, id, 'NX', 'PX', lease_ms)
-- lane B, P0 only
if not got and p0_lane == '1' and redis.call('ZSCORE', q, member) == '0' then
  got = redis.call('SET', active_p0, id, 'NX', 'PX', lease_ms)
end
```

A few details that matter in real life:

- A running request holds a lease that is renewed while the answer streams and released when it ends or the client disconnects. If a worker dies, the lease expires, so the GPU is never blocked forever.
- The queue is limited to 128 waiting requests and one spot is kept free for P0. Too many waiting requests give a 429; a queue timeout or a Redis problem gives a 503.
- The gate only applies to the local backends you list, so requests to remote providers do not take up the GPU lane.

The full implementation is [LiteLLM/gpu_admission.py](../LiteLLM/gpu_admission.py) and [LiteLLM/gpu_priority_hook.py](../LiteLLM/gpu_priority_hook.py).

### 4. Switch it on in LiteLLM

```yaml
litellm_settings:
  callbacks: ["priority_hook.proxy_handler_instance", "gpu_priority_hook.proxy_handler_instance"]
general_settings:
  coordination_redis:
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
```

Environment variables for the LiteLLM container:

```
GPU_PRIORITY_ENABLED=true
GPU_PRIORITY_ENDPOINTS=llama-swap:8080
GPU_PRIORITY_P0_EXECUTION_SLOTS=1
GPU_PRIORITY_P0_RESERVED_WAITERS=1
PRIORITY_BACKGROUND_MAX_TOKENS=64
REDIS_HOST=litellm-redis
```

- `GPU_PRIORITY_ENDPOINTS` is the host:port of your local backends, comma separated.
- `GPU_PRIORITY_P0_EXECUTION_SLOTS=0` removes the extra lane, so P0 queues like everyone else.
- `PRIORITY_BACKGROUND_MAX_TOKENS=0` removes the token cap.

Mount `priority_hook.py`, `gpu_priority_hook.py` and `gpu_admission.py` read-only into `/app` and restart LiteLLM.

### 5. Let the backend run two requests

Without this the second request just waits *inside* the backend and you are back where you started.

- **SGLang:** `--max-running-requests 2`. I left SGLang's own priority scheduling off, because the gate never sends more than two requests anyway.
- **vLLM:** `--max-num-seqs` of at least 2 (many setups already use a much higher value).
- **llama.cpp:** `--parallel 2` (the context is split between the slots).

On a unified-memory GB10, check `MemAvailable` before raising concurrency on a model that already uses most of your memory.

### 6. Measure it

Please do not just trust the config. My probe starts one long background request and then, at the same moment, one request per tier. It prints the time to first token and the order the first tokens arrived in.

```bash
export MODEL=default
for t in P0 P1 P2 P3 P4; do read -rsp "$t key: " v; echo; export ${t}_KEY="$v"; done
scripts/test-priority.sh --rounds 5 --p0-max-ttft 5 --blocker-tier P3 --load-tokens 256
```

`MODEL` is the model name or alias you really use. My results (first token, p50 over 5 rounds):

| Background request | P0 | P1 | P4 | Order |
|---|---|---|---|---|
| 256 tokens from P4, no P0 lane | 10.9 s | 11.7 s | 14.4 s | correct |
| 256 tokens from P3, with P0 lane | **0.35 s** | 9.7 s | 12.1 s | correct |
| P4 asks for 256, gets cut to 64 | 0.35 s | 1.4 s | 3.8 s | correct |

Five rounds on one model on one machine is a smoke test, not a proper latency study. Your numbers will depend on your model and your decode speed.

There is also [scripts/validate-priority.sh](../scripts/validate-priority.sh), which checks the ordering logic against a throwaway Redis, without any keys and without touching the GPU.

## Things that bit me

- **A body `priority` plus `coordination_redis`** gives intermittent 500s in LiteLLM 1.100.0. Strip it, as in the hook above.
- **Virtual keys usually cannot read their own info** (`/key/info` returns 403 for LLM-only keys). For the test script I verify the tier with the admin key via `POST /v2/key/info`.
- **`/model/info` does not show router aliases or the `api_base`.** To confirm that a request really went to your local backend, look at the `x-litellm-model-api-base` response header.
- **Memory floors:** my launchers refuse to start a model when `MemAvailable` is below their floor, instead of swapping and risking an OOM. Good for a shared box, confusing in the middle of a benchmark.

## Where this stops

If you need P1 to P3 to jump the queue as well, or you want to cancel a running request the moment a P0 arrives, this setup does not do that. The next step would be one lane per tier, or aborting the lower-priority stream at the gate and letting the caller retry. I have not built that.
