# litellm.Dockerfile
# Pin the signed stable release; floating tags were affected by the Oct 2026 incident.
FROM ghcr.io/berriai/litellm-database:v1.106.6@sha256:7ef0e78fe00ad5ea7ea45666cc1d1f02cb7076aee69091543bf397de52db33ce

# Local, in-process prompt compression callback. Provider credentials remain
# owned by LiteLLM; this is inert until an external alias is explicitly
# allowlisted through HEADROOM_EXTERNAL_MODELS.
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir "headroom-ai==0.37.0"
