"""Priority hook: key metadata.priority (higher = more important, 0..100) -> internal priority
100 - key priority (LOWER served first), used by the GPU gate and, for local llama-swap FP8 aliases,
forwarded as extra_body.priority. The request-body `priority` is removed: it would activate
LiteLLM's router scheduler, whose Redis queue raises TypeError (tuple vs list) under concurrency."""
from typing import Any, Optional
from urllib.parse import urlsplit

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # selftest without litellm
    CustomLogger = object


def body_priority(metadata: Any) -> Optional[int]:
    p = metadata.get("priority") if isinstance(metadata, dict) else None
    if isinstance(p, bool) or not isinstance(p, int):
        return None
    return 100 - max(0, min(100, p))


class PriorityHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Optional[dict]:
        data.pop("priority", None)  # never let a client or LiteLLM's buggy scheduler see it
        return data

    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: Any = None) -> Optional[dict]:
        model = kwargs.get("model", "")
        if not isinstance(model, str):
            return kwargs
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")
        if model not in ("Huihui-Qwen3.8-27B-abliterated-FP8", "dgx-spark-local"):
            return kwargs
        api_base = kwargs.get("api_base")
        if not isinstance(api_base, str):
            return kwargs
        try:
            backend = urlsplit(api_base)
        except ValueError:
            return kwargs
        if backend.scheme not in ("http", "https") or backend.hostname != "llama-swap":
            return kwargs
        request_metadata = kwargs.get("metadata")
        metadata = request_metadata.get("user_api_key_metadata") if isinstance(request_metadata, dict) else None
        p = body_priority(metadata)
        if p is None:
            return kwargs
        extra_body = kwargs.get("extra_body")
        if extra_body is None:
            extra_body = {}
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be an object")
        kwargs["extra_body"] = {**extra_body, "priority": p}
        return kwargs


proxy_handler_instance = PriorityHook()
