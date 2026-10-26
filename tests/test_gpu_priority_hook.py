"""Focused lifecycle self-check; python3 tests/test_gpu_priority_hook.py."""
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LiteLLM"))
import gpu_priority_hook as gpu


class Lease:
    def __init__(self):
        self.lost = asyncio.Event()
        self.releases = 0

    async def release(self):
        self.releases += 1


class Gate:
    def __init__(self):
        self.calls = []
        self.lease = Lease()

    async def acquire(self, priority):
        self.calls.append(priority)
        return self.lease


async def check():
    os.environ["GPU_PRIORITY_ENABLED"] = "true"
    os.environ["GPU_PRIORITY_ENDPOINTS"] = "llama-swap:8080,tts:8000"
    hook = gpu.proxy_handler_instance
    scope = {"type": "http"}
    async def receive():
        return {"type": "http.request", "body": b""}
    async def send(message):
        assert gpu._gate.lease.releases == 0
    for call in ("acompletion", "aspeech", "speech", "atranscription", "transcription", "text_to_speech", "audio_transcription"):
        gate = gpu._gate = Gate()
        async def app(scope, receive, send):
            await hook.async_pre_call_hook(SimpleNamespace(metadata={"priority": 100}), None, {"priority": 999}, call)
            kwargs = {"api_base": "http://tts:8000/v1"}
            await asyncio.gather(hook.async_pre_call_deployment_hook(kwargs, call), hook.async_pre_call_deployment_hook(kwargs, call))
            await hook.async_pre_call_deployment_hook({"api_base": "https://cloud.example/v1"}, call)
            await send({"type": "http.response.start", "status": 200})
            await asyncio.sleep(0)
            await send({"type": "http.response.body", "body": b"chunk", "more_body": True})
            await send({"type": "http.response.body", "body": b"end"})
        await gpu.GPUAdmissionMiddleware(app)(scope, receive, send)
        assert gate.calls == ([0] if call == "acompletion" else [])
        assert gate.lease.releases == (1 if call == "acompletion" else 0)
        assert gpu._request.get() is None
    for mode in ("error", "cancel", "lost"):
        gate = gpu._gate = Gate()
        started = asyncio.Event()
        async def app(scope, receive, send):
            await hook.async_pre_call_deployment_hook({"api_base": "http://tts:8000"}, "acompletion")
            started.set()
            if mode == "error":
                raise ValueError("test")
            await asyncio.Event().wait()
        task = asyncio.create_task(gpu.GPUAdmissionMiddleware(app)(scope, receive, send))
        await started.wait()
        if mode == "cancel":
            task.cancel()
        elif mode == "lost":
            gate.lease.lost.set()
        try:
            await task
        except (ValueError, asyncio.CancelledError, RuntimeError):
            pass
        else:
            raise AssertionError("error/cancel/loss was swallowed")
        assert gate.lease.releases == 1
    gate = gpu._gate = Gate()
    for base, call in (("http://tts.remote.example:8000", "aspeech"), ("http://tts:8000", "pass_through_endpoint")):
        await hook.async_pre_call_deployment_hook({"api_base": base}, call)
    assert gate.calls == []
    try:
        await hook.async_pre_call_deployment_hook({"api_base": "http://tts:8000"}, "acompletion")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing context admitted local work")
    # Speech must progress while the shared LLM gate cannot grant a lease.
    class BlockedGate(Gate):
        async def acquire(self, priority):
            await asyncio.Event().wait()
    gpu._gate = BlockedGate()
    for call in ("speech", "aspeech", "transcription", "atranscription"):
        await asyncio.wait_for(hook.async_pre_call_deployment_hook(
            {"api_base": "http://tts:8000", "stream": True}, SimpleNamespace(value=call)), .1)
    for failure in (False, True):
        from gpu_admission import AdmissionTimeout
        class SlowGate(Gate):
            async def acquire(self, priority):
                if failure:
                    raise AdmissionTimeout("test")
                return self.lease
        gpu._gate = SlowGate()
        async def app(scope, receive, send):
            with patch.object(gpu.asyncio, "get_running_loop") as loop:
                loop.return_value.time.side_effect = [0, 1.1]
                await hook.async_pre_call_deployment_hook({"api_base": "http://tts:8000"}, "acompletion")
        with patch.object(gpu.logger, "warning") as warning:
            try:
                await gpu.GPUAdmissionMiddleware(app)(scope, receive, send)
            except gpu.HTTPException as exc:
                assert failure and exc.status_code == 503
            else:
                assert not failure
            warning.assert_called_once_with("GPU admission wait %.3fs (priority=%s)", 1.1, 40)
    os.environ["GPU_PRIORITY_ENABLED"] = "false"
    await hook.async_pre_call_deployment_hook({"api_base": "http://tts:8000"}, "acompletion")


asyncio.run(check())
print("GPU priority lifecycle self-check passed")
