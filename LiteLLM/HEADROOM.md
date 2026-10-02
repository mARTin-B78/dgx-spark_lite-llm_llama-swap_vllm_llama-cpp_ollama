# LiteLLM Headroom compression

The OpenAI and Anthropic aliases use Headroom as a transparent upstream proxy,
so their traffic follows LiteLLM → Headroom → provider. Provider credentials
remain in LiteLLM and are forwarded only with the provider request.

Headroom runs in `token` mode to prioritize token reduction. `cache` mode was
tested first but preserved the complete first-request prefix, including a
14.8K-token synthetic tool result, and therefore recorded no compression.

The native LiteLLM `headroom-compression` pre-call guardrail is also registered
for testing and visibility in Cost Optimization. LiteLLM 1.100 currently makes
a duplicate guardrail pass through the LAN resolver and ignores `fail_open`, so
do not opt requests into that guardrail until LiteLLM fixes the duplicate pass.

The Portainer-managed Headroom stack reserves `172.24.0.4` on the private
`headroom_net`; the LiteLLM guardrail uses that address to avoid a resolver
collision with the DGX host name. Port 8787 remains host-loopback-only.

Headroom inbound token auth is intentionally disabled for this private proxy:
OpenAI and Anthropic credentials use the same authorization headers and must
pass through to their providers. Do not publish Headroom's port beyond
`127.0.0.1` without redesigning that credential boundary.

The guardrail is opt-in so local DGX models are unaffected. After upgrading
LiteLLM to a version without the duplicate-pass bug, enable it per
OpenAI-format request with:

```json
"guardrails": ["headroom-compression"]
```

For an Anthropic-format `/v1/messages` request, use:

```json
"litellm_metadata": {"guardrails": ["headroom-compression"]}
```

Successful responses include the header
`x-litellm-applied-guardrails: headroom-compression`.

## Rollback

Remove `headroom-compression` from the request, or set
`x-headroom-bypass: true` when the guardrail is attached through a key.
