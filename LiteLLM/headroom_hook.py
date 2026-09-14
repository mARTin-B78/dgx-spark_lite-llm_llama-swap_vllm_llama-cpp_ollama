"""Explicit external-model Headroom callback for the LiteLLM proxy.

No model is compressed unless its LiteLLM alias appears in
HEADROOM_EXTERNAL_MODELS. This deliberately keeps every DGX-local model on its
normal direct path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from headroom.integrations.litellm_callback import HeadroomCallback

logger = logging.getLogger(__name__)


def _external_models() -> frozenset[str]:
    return frozenset(
        model.strip()
        for model in os.environ.get("HEADROOM_EXTERNAL_MODELS", "").split(",")
        if model.strip()
    )


class ExternalOnlyHeadroomCallback(HeadroomCallback):
    """Compress calls only for explicitly approved external LiteLLM aliases."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict[Any, Any] | None = None,
        call_type: str = "",
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[Any, Any] | None:
        if isinstance(cache, dict) and isinstance(data, str):
            data, call_type = cache, data
        if data is None:
            return None

        model = str(data.get("model", ""))
        if model not in _external_models():
            return data

        logger.info("Headroom compression selected for external model=%s", model)
        return await super().async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type=call_type,
            *_args,
            **_kwargs,
        )


headroom_external_only_callback = ExternalOnlyHeadroomCallback()
