#!/usr/bin/env bash
# Deterministic checks against disposable Redis; no API keys or GPU required.
set -Eeuo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
redis_image=redis:7-alpine
python_image=ghcr.io/martin-b78/litellm-spark:latest
docker image inspect "$redis_image" "$python_image" >/dev/null
redis_id=$(docker run --pull never -d --rm --network none "$redis_image" redis-server --save '' --appendonly no)
cleanup() { docker stop "$redis_id" >/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --pull never --rm --network "container:$redis_id" \
  --entrypoint python -e PYTHONDONTWRITEBYTECODE=1 \
  -e GPU_ADMISSION_TEST_REDIS_URL=redis://127.0.0.1:6379/0 \
  -v "$repo/LiteLLM/gpu_admission.py:/work/LiteLLM/gpu_admission.py:ro" \
  -v "$repo/LiteLLM/gpu_priority_hook.py:/work/LiteLLM/gpu_priority_hook.py:ro" \
  -v "$repo/LiteLLM/priority_hook.py:/work/LiteLLM/priority_hook.py:ro" \
  -v "$repo/tests:/work/tests:ro" \
  -w /work "$python_image" -c '
import socket, subprocess, sys, time
for attempt in range(50):
    try:
        with socket.create_connection(("127.0.0.1", 6379), timeout=.2):
            break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit("FAIL: disposable Redis did not start")
for test in ("test_priority_hook.py", "test_gpu_priority_hook.py",
             "test_gpu_admission.py", "test_priority_integration.py"):
    print("Checking " + test, flush=True)
    subprocess.run([sys.executable, "tests/" + test], check=True, timeout=60)
print("PASS: P0-P4 metadata, admission order, reservation and stream lifecycle")
'
