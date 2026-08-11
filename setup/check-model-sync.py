#!/usr/bin/env python3
"""
Check that LiteLLM's published model names stay aligned with llama-swap.

This is a drift detector, not a rewriter:
- it compares the active model names in llama-swap/config.yaml
- against the model_name entries exposed by LiteLLM/config.yaml

The goal is to keep OpenWebUI reading from LiteLLM only, while llama-swap
remains the source of truth for which models actually exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_SWAP_CONFIG = REPO_ROOT / "llama-swap" / "config.yaml"
LITELLM_CONFIG = REPO_ROOT / "LiteLLM" / "config.yaml"

# These are intentionally present in LiteLLM but not mirrored 1:1 in llama-swap.
ALLOWED_LITELLM_ONLY = {
    "auto_router1",
    "semantic-router",
    "text-embedding-3-small",
    "Qwen3.5-4B-Q4_K_M-always-on",
    "claude-3-5-sonnet-20241022",
    "qwen-3-coder-30b-fp8",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def main() -> int:
    llama_swap = load_yaml(LLAMA_SWAP_CONFIG)
    litellm = load_yaml(LITELLM_CONFIG)

    source_models = set((llama_swap.get("models") or {}).keys())
    target_models = {
        entry.get("model_name")
        for entry in (litellm.get("model_list") or [])
        if isinstance(entry, dict) and entry.get("model_name")
    }

    missing = sorted(source_models - target_models)
    stale = sorted((target_models - source_models) - ALLOWED_LITELLM_ONLY)

    if not missing and not stale:
        print("Model sync OK")
        print(f"  llama-swap models: {len(source_models)}")
        print(f"  LiteLLM models:    {len(target_models)}")
        return 0

    if missing:
        print("Missing in LiteLLM (present in llama-swap):")
        for name in missing:
            print(f"  - {name}")

    if stale:
        print("Stale in LiteLLM (not present in llama-swap):")
        for name in stale:
            print(f"  - {name}")

    print()
    print("Fix: regenerate or edit LiteLLM/config.yaml so these names match llama-swap.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
