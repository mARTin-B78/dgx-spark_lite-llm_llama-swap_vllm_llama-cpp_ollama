"""
Anthropic OAuth Header Stripping Middleware for LiteLLM.

Aggressively strips 'Authorization' and 'x-api-key' headers containing
Anthropic OAuth tokens (e.g. 'sk-ant-oat-...') when requests fail over
from the Anthropic subscription pool (e.g. claude-code-pool) to non-Anthropic
providers (OpenRouter or local models).
"""

import logging
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("anthropic_header_strip")
logger.setLevel(logging.INFO)

print("[anthropic_header_strip] Initializing Anthropic OAuth Header Strip middleware...", flush=True)


class AnthropicOAuthHeaderStrip(CustomLogger):
    """
    Custom proxy hook to strip Anthropic OAuth tokens when routing/failing over
    to non-Anthropic providers.
    """

    def __init__(self):
        super().__init__()
        logger.info("AnthropicOAuthHeaderStrip initialized successfully")
        print("[anthropic_header_strip] AnthropicOAuthHeaderStrip initialized successfully", flush=True)

    @staticmethod
    def _is_anthropic_oauth_token(val: Any) -> bool:
        """Check if a string or header value contains an Anthropic OAuth token or key."""
        if not isinstance(val, str):
            return False
        v = val.strip().lower()
        return "sk-ant-oat" in v or "sk-ant-api" in v or "sk-ant-" in v

    @staticmethod
    def _is_anthropic_destination(data: dict) -> bool:
        """
        Check if the destination provider/model is Anthropic.
        Returns True if destination is Anthropic, False if non-Anthropic (OpenRouter or local).
        """
        # 1. Check custom_llm_provider
        provider = (
            data.get("custom_llm_provider")
            or data.get("litellm_params", {}).get("custom_llm_provider")
            or ""
        )
        if isinstance(provider, str) and provider.strip():
            p = provider.strip().lower()
            if p == "anthropic":
                return True
            if p in {"openrouter", "openai", "ollama", "custom_openai", "vllm", "llama.cpp", "sglang"}:
                return False

        # 2. Check model name
        model = (
            data.get("model")
            or data.get("litellm_params", {}).get("model")
            or ""
        )
        if isinstance(model, str) and model.strip():
            m = model.strip().lower()
            # Explicit anthropic models or pool
            if m.startswith("anthropic/") or m == "claude-code-pool":
                return True
            # Explicit non-anthropic models
            if (
                m.startswith("openrouter/")
                or m == "claude-api-backup"
                or m == "dgx-spark-local"
                or m.startswith("openai/")
                or m.startswith("ollama/")
                or "llama" in m
                or "qwen" in m
            ):
                return False

        # 3. Check api_base
        api_base = (
            data.get("api_base")
            or data.get("litellm_params", {}).get("api_base")
            or ""
        )
        if isinstance(api_base, str) and api_base.strip():
            ab = api_base.strip().lower()
            if "anthropic.com" in ab:
                return True
            if any(h in ab for h in ["openrouter", "llama-swap", "localhost", "127.0.0.1", "172.", "192.168."]):
                return False

        # 4. Fallback check for claude in model name without openrouter/backup
        if isinstance(model, str) and "claude" in model.lower() and not ("openrouter" in model.lower() or "backup" in model.lower()):
            return True

        # Default to non-anthropic if not matching Anthropic
        return False

    def _strip_headers_dict(self, headers: dict) -> list[str]:
        """Strip Authorization and x-api-key headers containing Anthropic tokens."""
        if not isinstance(headers, dict):
            return []
        removed = []
        for key in list(headers.keys()):
            k_norm = str(key).strip().lower().replace("_", "-")
            if k_norm in {"authorization", "x-api-key", "api-key"}:
                val = headers[key]
                if self._is_anthropic_oauth_token(val):
                    removed.append(str(key))
                    del headers[key]
        return removed

    def _process_and_strip(self, target: dict, call_type: Optional[str] = None) -> dict:
        """Inspect destination and strip Anthropic OAuth headers if non-Anthropic."""
        if not isinstance(target, dict):
            return target

        # If destination is Anthropic, preserve tokens
        if self._is_anthropic_destination(target):
            return target

        model_name = target.get("model") or target.get("litellm_params", {}).get("model") or "unknown"
        all_removed = []

        # 1. Strip from headers
        if "headers" in target and isinstance(target["headers"], dict):
            all_removed.extend(self._strip_headers_dict(target["headers"]))

        # 2. Strip from extra_headers
        if "extra_headers" in target and isinstance(target["extra_headers"], dict):
            all_removed.extend(self._strip_headers_dict(target["extra_headers"]))

        # 3. Strip from litellm_params headers / extra_headers
        litellm_params = target.get("litellm_params")
        if isinstance(litellm_params, dict):
            if "headers" in litellm_params and isinstance(litellm_params["headers"], dict):
                all_removed.extend(self._strip_headers_dict(litellm_params["headers"]))
            if "extra_headers" in litellm_params and isinstance(litellm_params["extra_headers"], dict):
                all_removed.extend(self._strip_headers_dict(litellm_params["extra_headers"]))
            if self._is_anthropic_oauth_token(litellm_params.get("api_key")):
                all_removed.append("litellm_params.api_key")
                del litellm_params["api_key"]

        # 4. Strip from metadata headers if present
        metadata = target.get("metadata")
        if isinstance(metadata, dict):
            if "headers" in metadata and isinstance(metadata["headers"], dict):
                all_removed.extend(self._strip_headers_dict(metadata["headers"]))
            if "extra_headers" in metadata and isinstance(metadata["extra_headers"], dict):
                all_removed.extend(self._strip_headers_dict(metadata["extra_headers"]))

        # 5. Check top-level api_key
        if self._is_anthropic_oauth_token(target.get("api_key")):
            all_removed.append("api_key")
            del target["api_key"]

        # 6. Check provider_specific_header
        psh = target.get("provider_specific_header")
        if psh:
            if isinstance(psh, dict):
                psh_headers = psh.get("extra_headers")
                if isinstance(psh_headers, dict):
                    all_removed.extend(self._strip_headers_dict(psh_headers))
            elif isinstance(psh, list):
                for item in psh:
                    if isinstance(item, dict):
                        psh_headers = item.get("extra_headers")
                        if isinstance(psh_headers, dict):
                            all_removed.extend(self._strip_headers_dict(psh_headers))

        if all_removed:
            logger.info(
                "[anthropic_header_strip] Stripped Anthropic OAuth tokens from %s for destination model=%s (call_type=%s)",
                all_removed,
                model_name,
                call_type,
            )
            print(
                f"[anthropic_header_strip] Stripped Anthropic OAuth tokens from {all_removed} for destination model={model_name}",
                flush=True,
            )

        return target

    async def async_pre_call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Pre-call logic checking destination provider and stripping Anthropic OAuth tokens.
        Accepts flexible signatures:
          async_pre_call(data, call_type=...)
          async_pre_call(user_api_key_dict, cache, data, call_type)
          async_pre_call(kwargs, call_type=...)
          async_pre_call(data=..., call_type=...)
        """
        target_dict = None
        call_type = kwargs.get("call_type")

        if args:
            if len(args) >= 3 and isinstance(args[2], dict):
                # (user_api_key_dict, cache, data, call_type)
                target_dict = args[2]
                if len(args) >= 4 and isinstance(args[3], str):
                    call_type = args[3]
            elif isinstance(args[0], dict):
                target_dict = args[0]
                if len(args) >= 2 and isinstance(args[1], str):
                    call_type = args[1]

        if target_dict is None:
            if "data" in kwargs and isinstance(kwargs["data"], dict):
                target_dict = kwargs["data"]
            elif "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
                target_dict = kwargs["kwargs"]
            elif kwargs:
                target_dict = kwargs

        if target_dict is None or not isinstance(target_dict, dict):
            return target_dict

        return self._process_and_strip(target_dict, call_type=call_type)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: Optional[dict] = None,
        call_type: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> Optional[dict]:
        """LiteLLM proxy pre-call hook callback."""
        if isinstance(cache, dict) and isinstance(data, str):
            data, call_type = cache, data
        if data is None:
            return None
        return await self.async_pre_call(data, call_type=call_type)

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, Any],
        call_type: Any = None,
    ) -> Optional[dict]:
        """LiteLLM deployment hook callback invoked before sending to provider/fallback."""
        if kwargs is None:
            return None
        return await self.async_pre_call(kwargs, call_type=str(call_type) if call_type else None)


# Instantiate the proxy handler for LiteLLM callbacks
proxy_handler_instance = AnthropicOAuthHeaderStrip()
proxy_handler = proxy_handler_instance
anthropic_header_strip_hook = proxy_handler_instance
