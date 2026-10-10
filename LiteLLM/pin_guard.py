"""Pin guard: while a model is pinned (LiteLLM/pin/pinned_model.txt), block deployments that go to
llama-swap for any other model, so llama-swap never evicts the pinned one. Remote/always-on pass.
Runs per deployment attempt (inside the router) so a raised error triggers the normal fallback chain."""
import os
from typing import Any, Optional

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # selftest without litellm
    CustomLogger = object

PIN_FILE = os.environ.get("PIN_FILE", "/app/pin/pinned_model.txt")
SWAP_MARKER = os.environ.get("PIN_SWAP_MARKER", "llama-swap")  # substring of api_base of swapped models
_cache = (None, "")  # (mtime, value)


def read_pin(path: str = PIN_FILE) -> str:
    global _cache
    try:
        m = os.stat(path).st_mtime_ns
        if m != _cache[0]:
            with open(path) as f:
                _cache = (m, f.read().strip())
        return _cache[1]
    except OSError:
        return ""


def blocked_reason(pin: str, model: str, api_base: str) -> Optional[str]:
    """Return message if request must be blocked, else None."""
    if not pin or SWAP_MARKER not in (api_base or ""):
        return None
    name = (model or "").split("/", 1)[-1] if (model or "").startswith("openai/") else (model or "")
    if name == pin or model == pin:
        return None
    return f"model {name} blocked: {pin} is pinned"


class PinGuard(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: Any = None) -> Optional[dict]:
        msg = blocked_reason(read_pin(), kwargs.get("model", ""), kwargs.get("api_base", ""))
        if msg:
            import litellm
            raise litellm.ServiceUnavailableError(message=msg, model=kwargs.get("model", ""), llm_provider="pin_guard")
        return kwargs


proxy_handler_instance = PinGuard()

if __name__ == "__main__":
    A = "http://llama-swap:8080/v1"
    assert blocked_reason("", "openai/X", A) is None                      # no pin
    assert blocked_reason("P", "openai/X", A) == "model X blocked: P is pinned"
    assert blocked_reason("P", "openai/P", A) is None                     # the pinned model
    assert blocked_reason("P", "openai/Y", "http://llama-qwen35-4b:19001/v1") is None  # always-on
    assert blocked_reason("P", "openrouter/qwen/q:free", "") is None      # remote
    print("pin_guard selftest ok")
