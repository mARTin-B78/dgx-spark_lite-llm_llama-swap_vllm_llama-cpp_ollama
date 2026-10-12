"""Hook -> real Redis -> simulated ASGI streams; no key DB, auth, or GPU.

Run with an explicit disposable GPU_ADMISSION_TEST_REDIS_URL.
"""
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

from redis.asyncio import Redis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LiteLLM"))
# Import before enabling: this harness owns its ASGI app, not proxy_server.
import gpu_priority_hook as gpu
from gpu_admission import GPUAdmission


async def check():
    client = Redis.from_url(os.environ["GPU_ADMISSION_TEST_REDIS_URL"],
                            socket_connect_timeout=2, socket_timeout=2)
    gate = GPUAdmission(client, key_prefix="test:priority-integration:" + uuid.uuid4().hex,
                        wait_timeout=15, lease_ttl=3, poll_interval=.01)
    previous_gate = gpu._gate
    previous_env = {name: os.environ.get(name) for name in
                    ("GPU_PRIORITY_ENABLED", "GPU_PRIORITY_ENDPOINTS")}
    gpu._gate = gate
    os.environ["GPU_PRIORITY_ENABLED"] = "true"
    os.environ["GPU_PRIORITY_ENDPOINTS"] = "llama-swap:8080"
    hook = gpu.GPUHook()
    tasks, entries, finals = [], [], []
    entered, finish = {}, {}
    active = 0

    async def wait_queue(scores):
        async def observed():
            while True:
                rows = await client.zrange(gate.keys[0], 0, -1, withscores=True)
                if [score for _, score in rows] == scores:
                    return
                await asyncio.sleep(.01)
        await asyncio.wait_for(observed(), 5)

    async def app(scope, receive, send):
        nonlocal active
        label = scope["label"]
        # Every caller forges highest urgency; only authenticated key metadata counts.
        data = {"priority": 0, "extra_body": {"priority": 0}}
        await hook.async_pre_call_hook(
            SimpleNamespace(metadata={"priority": scope["key_priority"]}),
            None, data, "acompletion")
        await hook.async_pre_call_deployment_hook(
            {"api_base": "http://llama-swap:8080/v1", **data}, "acompletion")
        state = gpu._request.get()
        assert state.priority == 100 - scope["key_priority"]
        assert active == 0, (label, entries, finals)
        active += 1
        entries.append(label)
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"chunk", "more_body": True})
            entered[label].set()
            await finish[label].wait()
            await send({"type": "http.response.body", "body": b"end", "more_body": False})
        finally:
            active -= 1

    middleware = gpu.GPUAdmissionMiddleware(app)

    def submit(label, key_priority):
        entered[label], finish[label] = asyncio.Event(), asyncio.Event()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            lease = gpu._request.get().lease
            assert active == 1
            assert await client.get(gate.keys[4]) == lease.ticket.encode()
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finals.append(label)

        task = asyncio.create_task(middleware(
            {"type": "http", "label": label, "key_priority": key_priority}, receive, send))
        tasks.append(task)
        return task

    try:
        blocker = submit("initial-P4", 20)
        await asyncio.wait_for(entered["initial-P4"].wait(), 5)
        for priority in range(4, -1, -1):
            submit("P" + str(priority), 100 - priority * 20)
        await wait_queue([0, 20, 40, 60, 80])
        assert entries == ["initial-P4"] and finals == []
        finish["initial-P4"].set()
        await asyncio.wait_for(blocker, 5)
        for priority in range(5):
            label = "P" + str(priority)
            await asyncio.wait_for(entered[label].wait(), 5)
            assert entries == ["initial-P4"] + ["P" + str(p) for p in range(priority + 1)]
            assert finals == entries[:-1], (entries, finals)
            assert active == 1
            await wait_queue([p * 20 for p in range(priority + 1, 5)])
            finish[label].set()
            await asyncio.wait_for(tasks[5 - priority], 5)
        assert finals == entries == ["initial-P4", "P0", "P1", "P2", "P3", "P4"]
        assert active == 0 and await client.get(gate.keys[4]) is None

        # Cancellation must remove both queued tickets and active stream leases.
        blocker = submit("cancel-active", 20)
        await asyncio.wait_for(entered["cancel-active"].wait(), 5)
        waiter = submit("cancel-queued", 100)
        await wait_queue([0])
        waiter.cancel()
        result = await asyncio.wait_for(asyncio.gather(waiter, return_exceptions=True), 5)
        assert isinstance(result[0], asyncio.CancelledError)
        await wait_queue([])
        assert "cancel-queued" not in entries
        blocker.cancel()
        result = await asyncio.wait_for(asyncio.gather(blocker, return_exceptions=True), 5)
        assert isinstance(result[0], asyncio.CancelledError)
        assert active == 0 and await client.get(gate.keys[4]) is None
        assert await client.hlen(gate.keys[2]) == 0
        assert await client.zcard(gate.keys[1]) == 0
        assert "cancel-active" not in finals
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        gpu._gate = previous_gate
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            await client.delete(*gate.keys)
        finally:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(check())
    print("Priority integration passed: key metadata P0..P4, stream lifetime, cancellation")
