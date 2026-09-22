#!/usr/bin/env bash
# GB10 host MemAvailable is visible in the llama-swap container.
# Conservative admission budget for the CURRENT HashK + NEXTN 262K recipe:
# 104 GiB model/runtime/startup allowance + 8 GiB system headroom.
# This is not a CUDA reservation; unrelated Docker launches can still race it.
set -euo pipefail

check_memory() {
    local available_kib="$1"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || return 2
    (( available_kib >= 111 * 1048576 ))
}

main() {
    local available_kib i
    exec 9>/tmp/flashnext-memory-guard.lock
    if ! flock -n 9; then
        echo '[flashnext-memory-guard] REFUSED: another guarded Flash-Next process is active.' >&2
        return 75
    fi
    # Require three sufficient samples. Never launch after a failed reading.
    for i in 1 2 3; do
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if ! check_memory "$available_kib"; then
            echo "[flashnext-memory-guard] REFUSED: MemAvailable=${available_kib:-unknown} KiB; at least 116736000 KiB (111 GiB) required for HashK/NEXTN/262K including 8 GiB headroom. Unload idle OCR/TTS/STT/ComfyUI models and retry. No model container was started." >&2
            return 75
        fi
        if (( i < 3 )); then sleep 2; fi
    done
    echo '[flashnext-memory-guard] PASS: three samples >=111 GiB available.' >&2
    if [[ "${1:-}" == --check-only ]]; then return 0; fi
    if (( $# == 0 )); then echo 'Missing launch command' >&2; return 64; fi
    # Keep lock held for the foreground docker run lifetime.
    "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
