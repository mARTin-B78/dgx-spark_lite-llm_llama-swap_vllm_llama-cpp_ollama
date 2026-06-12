"""
Complexity routing as a LiteLLM proxy pre-call hook.

Why: OpenClaw 2026.5.28 only calls the /v1/responses endpoint for its agent,
where LiteLLM's native `auto_router/complexity_router` model type is NOT
supported (returns 400 "Unmapped LLM provider ... custom_llm_provider=auto_router").
A CustomLogger.async_pre_call_hook, however, fires on EVERY endpoint
(chat/completions AND responses), so we do the complexity routing here by
rewriting data["model"] before LiteLLM routes the request.

It reuses LiteLLM's own ComplexityRouter.classify() (the exact same scoring the
`auto_router1` model used) — classify() does not touch the router instance, so we
can construct it with litellm_router_instance=None purely for scoring.

Trigger: any request whose model is in TRIGGER_MODELS gets rewritten to the
tier model. Everything else passes through untouched.
"""
from typing import Any, Optional, Tuple

from litellm.integrations.custom_logger import CustomLogger

try:
    from litellm.router_strategy.complexity_router.complexity_router import (
        ComplexityRouter,
    )
    _HAVE_CLASSIFIER = True
except Exception:  # pragma: no cover - fall back to default model if API moved
    _HAVE_CLASSIFIER = False

# --- Routing config (mirror of the old auto_router1 complexity_router_config) ---
ROUTER_CONFIG = {
    "tiers": {
        "SIMPLE": "Qwen3.5-4B-Q4_K_M",
        "MEDIUM": "Qwen3.5-4B-Q4_K_M",
        "COMPLEX": "Qwen3.5-4B-Q4_K_M",
        "REASONING": "Qwen3.6-27B-PrismaSCOUT-NVFP4",
    },
    "tier_boundaries": {
        "simple_medium": 0.18,
        "medium_complex": 0.38,
        "complex_reasoning": 0.62,
    },
    "token_thresholds": {"simple": 40, "complex": 500},
}
DEFAULT_MODEL = "Qwen3.5-4B-Q4_K_M"

# Models that should be complexity-routed. OpenClaw points its agent at
# "auto_router1"; "auto" is accepted as an alias too.
TRIGGER_MODELS = {"auto_router1", "auto"}

_classifier: Optional[Any] = None
if _HAVE_CLASSIFIER:
    try:
        _classifier = ComplexityRouter(
            model_name="auto_router1",
            litellm_router_instance=None,  # classify() never uses it
            complexity_router_config=ROUTER_CONFIG,
            default_model=DEFAULT_MODEL,
        )
    except Exception:
        _classifier = None


def _text_from_content(content: Any) -> str:
    """Flatten OpenAI content (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                # chat: {type:text, text:...}; responses: {type:input_text, text:...}
                t = p.get("text") or p.get("input_text") or ""
                if t:
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return ""


def _extract_prompt(data: dict) -> Tuple[str, str]:
    """Return (user_text, system_text) from either chat or responses payloads."""
    system_text = ""
    user_text = ""

    # chat/completions: messages[]
    msgs = data.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            txt = _text_from_content(m.get("content"))
            if role == "system":
                system_text += " " + txt
            elif role == "user":
                user_text = txt  # keep the latest user turn

    # responses: input can be a string or a list of message items
    inp = data.get("input")
    if isinstance(inp, str):
        if inp:
            user_text = inp
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            txt = _text_from_content(item.get("content"))
            if role == "system":
                system_text += " " + txt
            elif role == "user":
                user_text = txt

    # responses: top-level "instructions" acts as the system prompt
    instr = data.get("instructions")
    if isinstance(instr, str):
        system_text += " " + instr

    return user_text, system_text


class ComplexityResponsesRouter(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data: dict, call_type
    ):
        try:
            if not isinstance(data, dict):
                return data
            if data.get("model") not in TRIGGER_MODELS:
                return data

            chosen = DEFAULT_MODEL
            if _classifier is not None:
                user_text, system_text = _extract_prompt(data)
                tier, score, signals = _classifier.classify(
                    user_text or "", system_text or None
                )
                chosen = _classifier.get_model_for_tier(tier) or DEFAULT_MODEL
                try:
                    from litellm._logging import verbose_proxy_logger

                    verbose_proxy_logger.info(
                        "[complexity-hook] %s -> tier=%s score=%.3f model=%s",
                        call_type,
                        getattr(tier, "value", tier),
                        score,
                        chosen,
                    )
                except Exception:
                    pass

            data["model"] = chosen
        except Exception:
            # Never break the request; fall back to the always-hot default.
            data["model"] = DEFAULT_MODEL
        return data


# LiteLLM loads this instance via litellm_settings.callbacks
proxy_handler_instance = ComplexityResponsesRouter()
