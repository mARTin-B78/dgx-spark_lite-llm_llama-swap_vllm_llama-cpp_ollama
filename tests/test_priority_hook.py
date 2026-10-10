"""Self-check for LiteLLM/priority_hook.py (plain asserts; run: python3 tests/test_priority_hook.py)."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LiteLLM"))
from priority_hook import proxy_handler_instance as h


def run(meta, data):
    return asyncio.run(h.async_pre_call_hook(NS(metadata=meta), None, data, "completion"))


def deploy(meta, data):
    data["metadata"] = {"user_api_key_metadata": meta}
    return asyncio.run(h.async_pre_call_deployment_hook(data, "completion"))


assert run({"priority": 100}, {"priority": 99})["priority"] == 0      # overwrite client value
assert run({"priority": 20}, {})["priority"] == 80                    # P4 -> served last
assert run({}, {"priority": 5}) == {"priority": 5}                    # no metadata: untouched
assert "priority" not in run(None, {})
assert run({"priority": "high"}, {"priority": 5}) == {"priority": 5}  # non-int ignored
assert run({"priority": True}, {}) == {}
assert run({"priority": 500}, {})["priority"] == 0                    # clamp
assert run({"priority": -5}, {})["priority"] == 100
for model in ("Huihui-Qwen3.8-27B-abliterated-FP8", "dgx-spark-local"):
    data = {"model": model, "extra_body": {"priority": 999, "chat_template_kwargs": {"enable_thinking": False}}}
    result = run({"priority": 80}, data)
    assert result["priority"] == 20
    assert result["extra_body"]["priority"] == 999  # backend forwarding waits for resolved deployment
    base = "http://llama-swap:8080/v1"
    result = deploy({"priority": 80}, {"model": "openai/" + model, "api_base": base, "extra_body": {"priority": 999, "chat_template_kwargs": {"enable_thinking": False}}})
    assert result["extra_body"] == {"priority": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert deploy({"priority": 100}, {"model": model, "api_base": base})["extra_body"] == {"priority": 0}
    assert deploy({"priority": 20}, {"model": model, "api_base": base, "extra_body": None})["extra_body"] == {"priority": 80}
    for metadata in (None, "invalid", [], {}, {"priority": "high"}, {"priority": True}):
        original = {"model": model, "api_base": base, "extra_body": {"priority": 99}}
        result = deploy(metadata, original.copy())
        assert result["extra_body"] == original["extra_body"]
    for metadata in (None, "invalid", [], 1):
        original = {"model": model, "api_base": base, "extra_body": {"priority": 99}, "metadata": metadata}
        result = asyncio.run(h.async_pre_call_deployment_hook(original.copy(), "completion"))
        assert result["extra_body"] == original["extra_body"]
    for invalid in ([], "priority", 1, False):
        try:
            deploy({"priority": 80}, {"model": model, "api_base": base, "extra_body": invalid})
        except ValueError as error:
            assert str(error) == "extra_body must be an object"
        else:
            raise AssertionError("invalid extra_body accepted")
for model in ("openrouter/qwen/qwen3.8-27b:free", "Qwen3.5-4B-Q4_K_M-always-on"):
    original = {"model": model, "extra_body": {"priority": 99}}
    assert run({"priority": 80}, original.copy()) == {**original, "priority": 20}
    assert "extra_body" not in run({"priority": 80}, {"model": model})
    original = {"model": "openai/" + model, "api_base": "http://llama-swap:8080/v1", "extra_body": {"priority": 99}}
    assert deploy({"priority": 80}, original.copy())["extra_body"] == {"priority": 99}
assert deploy({"priority": 80}, {"model": "openai/dgx-spark-local", "api_base": "https://remote.example/v1", "extra_body": {"priority": 99}})["extra_body"] == {"priority": 99}
for base in ("https://llama-swap.remote.example/v1", "https://remote.example/llama-swap/v1", "http://llama-swap@remote.example/v1", "llama-swap", "http://[", None, 1):
    original = {"model": "openai/dgx-spark-local", "api_base": base, "extra_body": {"priority": 99}}
    assert deploy({"priority": 80}, original.copy())["extra_body"] == {"priority": 99}
for model in (None, 1, []):
    original = {"model": model, "api_base": "http://llama-swap:8080/v1"}
    assert "extra_body" not in deploy({"priority": 80}, original)
print("priority_hook selftest ok")
