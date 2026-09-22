#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="$repo_root/vllm/build/spark-vllm-docker"
build_after_update=false

if [[ "${1:-}" == "--build" ]]; then
    build_after_update=true
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: $0 [--build]" >&2
    exit 2
fi

if [[ ! -d "$upstream_dir/.git" ]]; then
    echo "ERROR: embedded upstream checkout not found: $upstream_dir" >&2
    exit 1
fi

git -C "$upstream_dir" fetch --quiet origin main
local_head="$(git -C "$upstream_dir" rev-parse HEAD)"
upstream_head="$(git -C "$upstream_dir" rev-parse origin/main)"

if [[ "$local_head" == "$upstream_head" ]]; then
    echo "vLLM builder is up to date ($local_head)."
    exit 0
fi

if ! git -C "$upstream_dir" diff --quiet HEAD || ! git -C "$upstream_dir" diff --cached --quiet; then
    echo "ERROR: vLLM builder has local changes; refusing to overwrite them." >&2
    git -C "$upstream_dir" status --short >&2
    exit 1
fi

if [[ -n "$(git -C "$upstream_dir" status --porcelain --untracked-files=all)" ]]; then
    echo "Note: preserving untracked local files in the vLLM builder checkout."
fi

read -r ahead behind < <(git -C "$upstream_dir" rev-list --left-right --count HEAD...origin/main)
if [[ "$ahead" != "0" ]]; then
    echo "ERROR: vLLM builder has $ahead local commit(s); refusing an automatic merge." >&2
    echo "       Review with: git -C '$upstream_dir' log --oneline HEAD...origin/main" >&2
    exit 1
fi

echo "Updating vLLM builder: $local_head -> $upstream_head"
git -C "$upstream_dir" merge --ff-only origin/main

if [[ "$build_after_update" == true ]]; then
    export GH_PAT="${GH_PAT:-$(gh auth token)}"
    "$repo_root/build_and_push.sh"
else
    echo "Source updated. Run '$repo_root/build_and_push.sh' to rebuild and publish images."
fi
