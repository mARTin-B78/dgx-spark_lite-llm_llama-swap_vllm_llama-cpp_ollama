#!/bin/bash
# wait_for_stable_memory: polls /proc/meminfo MemAvailable until two
# consecutive samples agree within MEM_STABLE_TOLERANCE_GIB, or
# MEM_STABLE_MAX_WAIT_S elapses.
#
# Why: docker container teardown (CUDA context release, page cache reclaim)
# is asynchronous — MemAvailable can keep rising for several seconds after
# `docker stop`/`docker rm` returns. If a launch script reads MemAvailable
# mid-reclaim, it undersizes gpu_memory_utilization (safe but wasteful) or,
# in the case of a genuinely overlapping load (e.g. an unload call that
# silently failed upstream), never sees the old model's memory freed at all.
# Waiting for two consecutive readings to agree is a cheap guard against
# both: it either confirms teardown has settled, or times out and proceeds
# with the last reading rather than hanging indefinitely.
wait_for_stable_memory() {
    local tolerance_gib="${MEM_STABLE_TOLERANCE_GIB:-0.5}"
    local max_wait_s="${MEM_STABLE_MAX_WAIT_S:-20}"
    local interval_s="${MEM_STABLE_INTERVAL_S:-2}"
    local elapsed=0
    local prev=""

    while [ "$elapsed" -lt "$max_wait_s" ]; do
        local cur_kb cur_gib
        cur_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
        cur_gib=$(awk -v k="$cur_kb" 'BEGIN{printf "%.2f", k/1048576}')

        if [ -n "$prev" ]; then
            local diff
            diff=$(awk -v a="$cur_gib" -v b="$prev" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')
            if awk -v d="$diff" -v t="$tolerance_gib" 'BEGIN{exit !(d<=t)}'; then
                echo "[mem-wait] MemAvailable stable at ${cur_gib}GiB (Δ${diff}GiB <= ${tolerance_gib}GiB tolerance)"
                return 0
            fi
            echo "[mem-wait] MemAvailable still moving: ${prev}GiB -> ${cur_gib}GiB (Δ${diff}GiB), waiting..."
        fi

        prev="$cur_gib"
        sleep "$interval_s"
        elapsed=$((elapsed + interval_s))
    done

    echo "[mem-wait] gave up waiting for stable memory after ${max_wait_s}s — proceeding with last reading (${prev}GiB)"
}
