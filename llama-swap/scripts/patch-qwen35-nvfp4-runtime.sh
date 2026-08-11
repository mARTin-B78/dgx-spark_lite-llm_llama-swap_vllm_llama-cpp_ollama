#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
MODEL_DIR="$SITE_PACKAGES/vllm/model_executor/models"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[qwen3.5-nvfp4] applying runtime patch set"

curl -fsSL \
    https://raw.githubusercontent.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4/master/qwen3_5_vl_moe.py \
    -o "$TMP_DIR/qwen3_5_vl_moe.py"
cp "$TMP_DIR/qwen3_5_vl_moe.py" "$MODEL_DIR/qwen3_5_vl_moe.py"

python3 - <<'PY'
from pathlib import Path
import re

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_vl_moe.py")
src = path.read_text()

pattern = re.compile(
    r'    def get_expert_mapping\(self\) -> list\[tuple\[str, str, int, str\]\]:\n'
    r'        from vllm\.model_executor\.layers\.fused_moe import SharedFusedMoE\n'
    r'        return SharedFusedMoE\.make_expert_params_mapping\(\n'
    r'            self,\n'
    r'            ckpt_gate_proj_name="gate_proj",\n'
    r'            ckpt_down_proj_name="down_proj",\n'
    r'            ckpt_up_proj_name="up_proj",\n'
    r'            num_experts=self\.config\.num_experts,\n'
    r'            num_redundant_experts=self\.num_redundant_experts,\n'
    r'        \)\n',
    re.M,
)
new = '''    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        from vllm.model_executor.layers.fused_moe import (
            fused_moe_make_expert_params_mapping,
        )

        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )
'''

if "SharedFusedMoE" in src and pattern.search(src):
    src = pattern.sub(new, src, count=1)
    path.write_text(src)
    print("qwen3_5_vl_moe.py: compatibility fallback applied for fused_moe_make_expert_params_mapping.")
else:
    print("qwen3_5_vl_moe.py: shared-fused expert mapping already compatible or anchor missing.")
PY

python3 - <<'PY'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
with open(path) as f:
    src = f.read()

old = (
    '    "Qwen3_5MoeForConditionalGeneration": (\n'
    '        "qwen3_5",\n'
    '        "Qwen3_5MoeForConditionalGeneration",\n'
    '    ),\n'
)
new = (
    '    "Qwen3_5MoeForConditionalGeneration": (\n'
    '        "qwen3_5_vl_moe",\n'
    '        "Qwen3_5MoeForConditionalGeneration",\n'
    '    ),\n'
)

if "qwen3_5_vl_moe" in src:
    print("registry.py: already pointing to qwen3_5_vl_moe, skipping.")
elif old not in src:
    print("ERROR: Qwen3_5MoeForConditionalGeneration entry not found in registry.py", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("registry.py: Qwen3_5MoeForConditionalGeneration redirected to qwen3_5_vl_moe.")
PY

python3 - <<'PY'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_triton_allocator_fix]"
if MARKER in src:
    print("qwen3_next.py: GDN Triton allocator fix already applied, skipping.")
    sys.exit(0)

old_logger = 'logger = init_logger(__name__)\n\nKVCache = tuple[torch.Tensor, torch.Tensor]'
new_logger = (
    'logger = init_logger(__name__)\n\n'
    + MARKER + '\n'
    '# Triton FLA kernels need a runtime memory allocator for global scratch space.\n'
    '# vLLM sets this in matmul_ogs.py for MoE, but GDN layers also need it.\n'
    'class _GDNTorchCudaAllocator:\n'
    '    """Torch-backed CUDA allocator for Triton FLA kernel scratch buffers."""\n'
    '    def __call__(self, size: int, alignment: int, stream=None) -> torch.Tensor:\n'
    '        return torch.empty(size, dtype=torch.uint8, device="cuda")\n'
    '\n'
    '_gdn_triton_allocator = _GDNTorchCudaAllocator()\n'
    '\n'
    'KVCache = tuple[torch.Tensor, torch.Tensor]'
)

if old_logger not in src:
    print("ERROR: logger anchor not found in qwen3_next.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_logger, new_logger, 1)

old_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        forward_context = get_forward_context()'
)
new_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        # Set Triton allocator for FLA kernels that need global scratch memory.\n'
    '        triton.set_allocator(_gdn_triton_allocator)\n'
    '        forward_context = get_forward_context()'
)

if old_fcore not in src:
    print("qwen3_next.py: allocator anchor not found, skipping allocator fix.")
else:
    src = src.replace(old_fcore, new_fcore, 1)

with open(path, "w") as f:
    f.write(src)
print("qwen3_next.py allocator patch step complete.")
PY

python3 - <<'PY'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_nan_guard]"
if MARKER in src:
    print("qwen3_next.py: NaN guard already applied, skipping.")
    sys.exit(0)

old_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        if self.layer_scale:'
)
new_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        ' + MARKER + '\n'
    '        # NVFP4 CUTLASS MoE kernel can produce NaN during prefill with\n'
    '        # GDN-derived activations. Only guard linear_attention (GDN) layers.\n'
    '        if self.layer_type == "linear_attention":\n'
    '            hidden_states = hidden_states.nan_to_num(nan=0.0)\n'
    '\n'
    '        if self.layer_scale:'
)

if old_mlp not in src:
    print("ERROR: MLP output anchor not found in Qwen3NextDecoderLayer.forward()", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_mlp, new_mlp, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied NaN guard (GDN layers only) to Qwen3NextDecoderLayer.forward() in qwen3_next.py.")
PY

echo "[qwen3.5-nvfp4] runtime patch complete"
