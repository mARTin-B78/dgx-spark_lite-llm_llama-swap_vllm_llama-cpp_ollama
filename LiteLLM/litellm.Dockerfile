# litellm.Dockerfile
# We pull from their stable release to ensure local reliability
FROM docker.litellm.ai/berriai/litellm-database:main-stable

# Local, in-process prompt compression callback. Provider credentials remain
# owned by LiteLLM; this is inert until an external alias is explicitly
# allowlisted through HEADROOM_EXTERNAL_MODELS.
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir "headroom-ai==0.37.0"
