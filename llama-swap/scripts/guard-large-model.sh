#!/usr/bin/env bash
# Serializes these stack-managed large models and watches only its own container.
# Does not reserve RAM or control manually launched external GPU workloads.
set -euo pipefail
name=${1:?container name required}
required=${2:?minimum available GiB required}
shift 2
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ && "$required" =~ ^[0-9]+$ ]] || exit 64
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/runtime"
exec 9>"$root/runtime/large-model.lock"
flock -n 9 || { echo '[memory-guard] REFUSED: another stack-managed large model is active.' >&2; exit 75; }
read_available() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
for sample in 1 2 3; do
    available=$(read_available)
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < required * 1048576 )); then
        echo "[memory-guard] REFUSED $name: available=${available:-unknown} KiB; required=${required} GiB including reserve. Unload idle GPU models, then retry. Nothing started." >&2
        exit 75
    fi
    (( sample == 3 )) || sleep 2
done
[[ "${1:-}" != --check-only ]] || exit 0
# Use the running compose service ID, independent of the user's container rename.
mapfile -t routers < <(docker ps -q --filter label=com.docker.compose.service=llama-swap)
(( ${#routers[@]} == 1 )) || { echo '[memory-guard] Expected one running llama-swap compose service.' >&2; exit 69; }
cidfile=$(mktemp "$root/runtime/container.XXXXXX")
rm "$cidfile" # docker requires a nonexistent file
runner=''
watcher=''
stop_owned() {
    local cid
    if [[ -s "$cidfile" ]]; then
        cid=$(cat "$cidfile")
        if [[ "$cid" =~ ^[a-f0-9]{64}$ ]]; then
            docker kill "$cid" >/dev/null 2>&1 || true
        fi
    fi
}
cleanup() {
    [[ -z "$watcher" ]] || kill "$watcher" 2>/dev/null || true
    stop_owned
    [[ -z "$runner" ]] || wait "$runner" 2>/dev/null || true
    rm -f "$cidfile"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP
echo "[memory-guard] Starting $name; watchdog floor 10 GiB available." >&2
docker run --rm --name "$name" --cidfile "$cidfile" \
    --network "container:${routers[0]}" --device nvidia.com/gpu=all "$@" > >(tee "$root/runtime/$name.log") 2>&1 &
runner=$!
(
    while kill -0 "$runner" 2>/dev/null; do
        available=$(read_available)
        if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < 10 * 1048576 )); then
            echo "[memory-guard] EMERGENCY STOP $name: available=${available:-unknown} KiB, floor=10 GiB." >&2
            # CID may not exist yet during docker create; keep watching until it does.
            if [[ -s "$cidfile" ]]; then stop_owned; exit 0; fi
        fi
        sleep 1
    done
) &
watcher=$!
set +e
wait "$runner"
result=$?
set -e
exit "$result"
