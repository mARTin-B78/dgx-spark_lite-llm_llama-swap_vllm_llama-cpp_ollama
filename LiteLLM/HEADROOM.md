# External-only Headroom compression

This LiteLLM image includes Headroom `0.37.0` as an in-process callback.
It does not add a second proxy, duplicate provider credentials, or affect
local DGX models.

## Enable a canary

1. Add one external provider model to `LiteLLM/config.yaml` using a unique,
   human-readable LiteLLM alias.
2. Put that provider's credential in the ignored `.env` file and pass it to
   the LiteLLM service in `docker-compose.yml`; never place it in YAML or Git.
3. Add only that alias to `HEADROOM_EXTERNAL_MODELS` in `.env`.
4. Build and deploy the `headroom-canary` image, then check logs for
   `Headroom compression selected`.

With an empty `HEADROOM_EXTERNAL_MODELS`, the callback is intentionally a
no-op. Keep every local DGX alias out of the allowlist.

## Rollback

Set `HEADROOM_EXTERNAL_MODELS=` and redeploy, or return the LiteLLM service to
its prior image. Neither action changes models, database data, or credentials.
