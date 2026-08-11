#!/usr/bin/env bash
# Kills ds4-server immediately if system memory gets dangerously low.
# GB10 unified memory means the desktop/display GPU allocations share the same
# pool as ds4-server's CUDA memory — if that pool gets too tight, the GPU driver
# itself can destabilize and crash the whole graphical session (observed:
# repeated "NVRM: Out of memory" in dmesg immediately followed by VS Code dying).
# This polls fast (every 2s) so it can react before a slow squeeze turns into that.
set -euo pipefail

THRESHOLD_GIB="${THRESHOLD_GIB:-8}"
LOG="/home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/logs/ds4-watchdog.log"

echo "$(date '+%F %T') watchdog started, threshold=${THRESHOLD_GIB}GiB" >> "$LOG"

while true; do
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  avail_gib=$(awk -v k="$avail_kb" 'BEGIN{printf "%.2f", k/1048576}')
  pid=$(pgrep -f "ds4/ds4-server" | head -1 || true)

  if [ -z "$pid" ]; then
    # ds4-server not running, nothing to guard
    sleep 5
    continue
  fi

  if awk -v a="$avail_gib" -v t="$THRESHOLD_GIB" 'BEGIN{exit !(a < t)}'; then
    echo "$(date '+%F %T') MemAvailable=${avail_gib}GiB < ${THRESHOLD_GIB}GiB -- KILLING ds4-server pid=$pid" >> "$LOG"
    kill -9 "$pid" 2>/dev/null || true
    echo "$(date '+%F %T') killed. watchdog continuing to idle-watch." >> "$LOG"
  fi

  sleep 2
done
