#!/usr/bin/env python3
"""
Test suite for Anthropic OAuth Header Stripping Middleware and Fallback Routing.
"""

import asyncio
import sys
import unittest
from pathlib import Path

# Add LiteLLM directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LiteLLM"))

from anthropic_header_strip import (
    AnthropicOAuthHeaderStrip,
    proxy_handler_instance,
)


class TestAnthropicOAuthHeaderStrip(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hook = proxy_handler_instance

    async def test_anthropic_destination_preserves_oauth_tokens(self):
        """Requests destined for Anthropic should keep their OAuth tokens."""
        req_pool = {
            "model": "claude-code-pool",
            "headers": {
                "Authorization": "Bearer sk-ant-oat-token-abc12345",
                "x-api-key": "sk-ant-oat-token-abc12345",
                "Content-Type": "application/json",
            },
            "extra_headers": {
                "Authorization": "Bearer sk-ant-oat-token-abc12345",
            },
            "api_key": "sk-ant-oat-token-abc12345",
        }
        res = await self.hook.async_pre_call(req_pool)
        self.assertEqual(res["headers"]["Authorization"], "Bearer sk-ant-oat-token-abc12345")
        self.assertEqual(res["headers"]["x-api-key"], "sk-ant-oat-token-abc12345")
        self.assertEqual(res["headers"]["Content-Type"], "application/json")
        self.assertEqual(res["extra_headers"]["Authorization"], "Bearer sk-ant-oat-token-abc12345")
        self.assertEqual(res["api_key"], "sk-ant-oat-token-abc12345")

    async def test_anthropic_provider_preserves_oauth_tokens(self):
        """Requests with custom_llm_provider='anthropic' should keep their OAuth tokens."""
        req = {
            "model": "anthropic/claude-3-7-sonnet",
            "custom_llm_provider": "anthropic",
            "headers": {
                "Authorization": "Bearer sk-ant-oat-subscription-key",
            },
        }
        res = await self.hook.async_pre_call(req)
        self.assertEqual(res["headers"]["Authorization"], "Bearer sk-ant-oat-subscription-key")

    async def test_fallback_to_openrouter_strips_oauth_tokens(self):
        """When failing over to OpenRouter, Anthropic OAuth headers and keys must be stripped."""
        req = {
            "model": "claude-api-backup",
            "custom_llm_provider": "openrouter",
            "headers": {
                "Authorization": "Bearer sk-ant-oat-token-12345",
                "x-api-key": "sk-ant-oat-token-12345",
                "User-Agent": "Claude-Code/1.0",
                "HTTP-Referer": "https://example.com",
            },
            "extra_headers": {
                "Authorization": "Bearer sk-ant-oat-token-12345",
            },
            "api_key": "sk-ant-oat-token-12345",
            "litellm_params": {
                "api_key": "sk-ant-oat-token-12345",
                "headers": {"x-api-key": "sk-ant-oat-token-12345"},
            },
            "provider_specific_header": {
                "extra_headers": {"Authorization": "Bearer sk-ant-oat-token-12345"}
            },
        }
        res = await self.hook.async_pre_call(req)
        # Verify OAuth headers are stripped
        self.assertNotIn("Authorization", res["headers"])
        self.assertNotIn("x-api-key", res["headers"])
        self.assertNotIn("Authorization", res["extra_headers"])
        self.assertNotIn("api_key", res)
        self.assertNotIn("api_key", res["litellm_params"])
        self.assertNotIn("x-api-key", res["litellm_params"]["headers"])
        self.assertNotIn("Authorization", res["provider_specific_header"]["extra_headers"])
        # Verify non-Anthropic headers are preserved
        self.assertEqual(res["headers"]["User-Agent"], "Claude-Code/1.0")
        self.assertEqual(res["headers"]["HTTP-Referer"], "https://example.com")

    async def test_fallback_to_local_llama_swap_strips_oauth_tokens(self):
        """When failing over to local llama-swap, Anthropic OAuth headers must be stripped."""
        req = {
            "model": "dgx-spark-local",
            "custom_llm_provider": "openai",
            "api_base": "http://llama-swap:8080/v1",
            "headers": {
                "Authorization": "Bearer sk-ant-oat-my-oauth-token",
                "X-Api-Key": "sk-ant-oat-my-oauth-token",
            },
            "extra_headers": {
                "authorization": "Bearer sk-ant-oat-my-oauth-token",
            },
        }
        res = await self.hook.async_pre_call(req)
        self.assertNotIn("Authorization", res["headers"])
        self.assertNotIn("X-Api-Key", res["headers"])
        self.assertNotIn("authorization", res.get("extra_headers", {}))

    async def test_non_oauth_headers_unaffected(self):
        """Standard provider keys or non-Anthropic tokens should NOT be stripped."""
        req = {
            "model": "claude-api-backup",
            "custom_llm_provider": "openrouter",
            "headers": {
                "Authorization": "Bearer sk-or-v1-real-openrouter-key",
                "Content-Type": "application/json",
            },
        }
        res = await self.hook.async_pre_call(req)
        self.assertEqual(res["headers"]["Authorization"], "Bearer sk-or-v1-real-openrouter-key")
        self.assertEqual(res["headers"]["Content-Type"], "application/json")

    async def test_async_pre_call_hook_interface(self):
        """Test proxy hook callback interface async_pre_call_hook."""
        req = {
            "model": "claude-api-backup",
            "headers": {"Authorization": "Bearer sk-ant-oat-test"},
        }
        res = await self.hook.async_pre_call_hook(None, None, req, "completion")
        self.assertNotIn("Authorization", res["headers"])

    async def test_async_pre_call_deployment_hook_interface(self):
        """Test deployment hook callback interface async_pre_call_deployment_hook."""
        req = {
            "model": "dgx-spark-local",
            "headers": {"x-api-key": "sk-ant-oat-test"},
        }
        res = await self.hook.async_pre_call_deployment_hook(req, "completion")
        self.assertNotIn("x-api-key", res["headers"])


if __name__ == "__main__":
    unittest.main()
