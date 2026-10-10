#!/usr/bin/env bash
# pin-model.sh pin <model> | unpin | status  -- writes LiteLLM/pin/pinned_model.txt (read by pin_guard.py, no restart needed)
set -Eeuo pipefail
f="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/LiteLLM/pin/pinned_model.txt"
case "${1:-}" in
  pin) [[ -n "${2:-}" && "$2" =~ ^[A-Za-z0-9._:/@+-]+$ ]] || { echo "usage: $0 pin <model>" >&2; exit 2; }
       printf '%s\n' "$2" > "$f.tmp" && mv "$f.tmp" "$f" ;;
  unpin) : > "$f.tmp" && mv "$f.tmp" "$f" ;;
  status) echo "pinned: $(tr -d '[:space:]' < "$f" 2>/dev/null || true)"; exit 0 ;;
  *) echo "usage: $0 pin <model> | unpin | status" >&2; exit 2 ;;
esac
echo "pinned: $(tr -d '[:space:]' < "$f")"
