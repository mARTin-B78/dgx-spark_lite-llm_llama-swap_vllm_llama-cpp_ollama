#!/usr/bin/env bash
# Restore LiteLLM alias default -> Qwen3.8-27B-NVFP4-DFlash2 (then: docker compose restart litellm)
set -Eeuo pipefail
docker exec litellm-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "update \"LiteLLM_Config\" set param_value=jsonb_set(param_value,'"'"'{model_group_alias,default}'"'"','"'"'\"Qwen3.8-27B-NVFP4-DFlash2\"'"'"') where param_name='"'"'router_settings'"'"'"'
echo "now: docker compose restart litellm"
