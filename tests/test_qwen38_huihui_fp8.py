"""Capture launcher arguments without Docker, GPU, or shared-service changes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

source = Path(__file__).resolve().parents[1] / "llama-swap/scripts/launch-qwen38-huihui-fp8.sh"
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    launcher = root / source.name
    shutil.copyfile(source, launcher)
    (root / "guard-large-model.sh").write_text(
        "#!/bin/bash\npython3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\"\n"
    )
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    args = json.loads(subprocess.check_output(["bash", str(launcher), "8123", "127.0.0.1"], text=True))
    assert args[:2] == ["qwen38-huihui-fp8-8123", "50"]
    for flag, value in {"--served-model-name": "Huihui-Qwen3.8-27B-abliterated-FP8", "--quantization": "fp8", "--dtype": "bfloat16", "--port": "8123", "--host": "127.0.0.1", "--context-length": "65536", "--max-total-tokens": "65536", "--max-running-requests": "1"}.items():
        assert args[args.index(flag) + 1] == value, flag
    assert any("munekazu/Huihui-Qwen3.8-27B-abliterated-FP8:/models/qwen38:ro" in arg for arg in args)
    assert "sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3" in args
    assert not any(arg.startswith("--speculative") for arg in args)
    cancellation_patch = args[args.index("-c") + 1]
    compile(cancellation_patch, "launcher cancellation patch", "exec")
    assert cancellation_patch.index('self.abort_request(rid)') < cancellation_patch.index('logger.exception')
    assert 'source.count(before) != 1' in cancellation_patch
    assert "--enable-priority-scheduling" in args
    assert "--schedule-low-priority-values-first" in args
    assert "--disable-priority-preemption" in args
    preempt_args = json.loads(subprocess.check_output(
        ["bash", str(launcher), "8123"], text=True,
        env={**os.environ, "QWEN38_PRIORITY_PREEMPTION": "1"},
    ))
    assert "--disable-priority-preemption" not in preempt_args
    assert subprocess.run(["bash", str(launcher)], capture_output=True).returncode != 0
print("Huihui FP8 launcher argument checks passed")
