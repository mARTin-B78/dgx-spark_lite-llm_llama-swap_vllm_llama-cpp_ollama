"""Priority hook: key metadata.priority (higher = more important, 0..100) -> internal priority
100 - key priority (LOWER served first), used by the GPU gate and, for local llama-swap FP8 aliases,
forwarded as extra_body.priority. Background keys (priority <= 20) are limited to
PRIORITY_BACKGROUND_MAX_TOKENS (default 64) per request. The request-body `priority` is removed: it would activate
LiteLLM's router scheduler, whose Redis queue raises TypeError (tuple vs list) under concurrency."""
import os
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


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


class PriorityHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Optional[dict]:
        data.pop("priority", None)  # never let a client or LiteLLM's buggy scheduler see it
        # Background keys get short generations so they cannot hold the GPU slot for long.
        cap = _int_env("PRIORITY_BACKGROUND_MAX_TOKENS", 64)
        metadata = getattr(user_api_key_dict, "metadata", None)
        key_priority = metadata.get("priority") if isinstance(metadata, dict) else None
        if (cap and "messages" in data and type(key_priority) is int
                and key_priority <= _int_env("PRIORITY_BACKGROUND_MAX_PRIORITY", 20)):
            for field in ("max_tokens", "max_completion_tokens"):
                value = data.get(field)
                if type(value) is int and value > cap:
                    data[field] = cap
            if "max_tokens" not in data and "max_completion_tokens" not in data:
                data["max_tokens"] = cap
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
