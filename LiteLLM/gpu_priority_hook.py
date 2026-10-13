"""Request-lifetime admission for configured local inference deployments."""
import asyncio
from contextvars import ContextVar
from importlib.metadata import version
import os
from urllib.parse import urlsplit
from fastapi import HTTPException

from priority_hook import CustomLogger, body_priority

_request = ContextVar("gpu_admission_request", default=None)
_gate = None
_INFERENCE = {"completion", "acompletion", "text_completion", "atext_completion", "embedding", "aembedding", "speech", "aspeech", "transcription", "atranscription", "image_generation", "aimage_generation", "responses", "aresponses"}


def enabled():
    return os.getenv("GPU_PRIORITY_ENABLED", "false").lower() in ("1", "true")


def endpoint(url):
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return None
        return parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


def configured_endpoints():
    entries = os.getenv("GPU_PRIORITY_ENDPOINTS", "").split(",")
    result = set()
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        parsed = endpoint("http://" + entry)
        if parsed is None or "/" in entry or "?" in entry or "#" in entry:
            raise RuntimeError("Invalid GPU_PRIORITY_ENDPOINTS")
        result.add(parsed)
    if not result:
        raise RuntimeError("GPU_PRIORITY_ENDPOINTS is required")
    return result


def admission():
    global _gate
    if _gate is None:
        from redis.asyncio import Redis
        from gpu_admission import GPUAdmission
        client = Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), password=os.getenv("REDIS_PASSWORD") or None, socket_connect_timeout=2, socket_timeout=2)
        _gate = GPUAdmission(
            client,
            reserved_p0_waiters=int(os.getenv("GPU_PRIORITY_P0_RESERVED_WAITERS", "1")),
            p0_execution_slots=int(os.getenv("GPU_PRIORITY_P0_EXECUTION_SLOTS", "1")))
    return _gate


class RequestState:
    def __init__(self):
        self.priority = 40
        self.lease = None
        self.lock = asyncio.Lock()
        self.task = None
        self.watcher = None

    async def watch(self):
        await self.lease.lost.wait()
        self.task.cancel()


class GPUAdmissionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            return await self.app(scope, receive, send)
        state = RequestState()
        token = _request.set(state)
        try:
            state.task = asyncio.create_task(self.app(scope, receive, send))
            try:
                await state.task
            except asyncio.CancelledError:
                if state.lease is not None and state.lease.lost.is_set():
                    raise RuntimeError("Local GPU admission lease lost") from None
                raise
        finally:
            if state.watcher is not None:
                state.watcher.cancel()
                await asyncio.gather(state.watcher, return_exceptions=True)
            if state.lease is not None:
                cleanup = asyncio.create_task(state.lease.release())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
            _request.reset(token)


class GPUHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        state = _request.get()
        if state is not None:
            priority = body_priority(getattr(user_api_key_dict, "metadata", None))
            state.priority = 40 if priority is None else priority
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type=None):
        call_type = getattr(call_type, "value", call_type)
        if not enabled() or call_type not in _INFERENCE or endpoint(kwargs.get("api_base")) not in configured_endpoints():
            return kwargs
        state = _request.get()
        if state is None:
            raise RuntimeError("Local GPU admission request context missing")
        async with state.lock:
            if state.lease is None:
                from gpu_admission import AdmissionError, AdmissionQueueFull
                try:
                    state.lease = await admission().acquire(state.priority)
                except AdmissionQueueFull:
                    raise HTTPException(429, "Local GPU admission queue full") from None
                except AdmissionError:
                    raise HTTPException(503, "Local GPU admission unavailable") from None
                state.watcher = asyncio.create_task(state.watch())
            if state.lease.lost.is_set():
                raise RuntimeError("Local GPU admission lease lost")
        return kwargs


def install_middleware():
    if not enabled():
        return
    if version("litellm") != "1.100.0":
        raise RuntimeError("GPU admission requires verified LiteLLM 1.100.0")
    configured_endpoints()
    from litellm.proxy.proxy_server import app
    if app.middleware_stack is None:
        raise RuntimeError("LiteLLM middleware stack unavailable")
    if not isinstance(app.middleware_stack, GPUAdmissionMiddleware):
        app.middleware_stack = GPUAdmissionMiddleware(app.middleware_stack)
        print("[gpu_priority_hook] GPU admission middleware installed", flush=True)


proxy_handler_instance = GPUHook()

# Also install at callback import: LITELLM_WORKER_STARTUP_HOOKS alone left requests unwrapped
# (idempotent; ponytail: remove once the startup-hook path is confirmed working).
try:
    install_middleware()
except ImportError as exc:  # proxy_server not importable yet (tests, early import)
    print(f"[gpu_priority_hook] middleware install deferred: {exc}", flush=True)
