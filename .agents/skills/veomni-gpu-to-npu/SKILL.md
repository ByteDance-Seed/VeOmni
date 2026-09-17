---
name: veomni-gpu-to-npu
description: "Use this skill when migrating a VeOmni GPU model to Ascend NPU. The model must already train on VeOmni GPU (patchgen-generated GPU path exists). Covers: CUDA/GPU-specific dependency scanning, NPU backend selection (rms_norm / rotary / cross_entropy / attention / moe / swiglu), device-agnostic API verification, NPU patchgen config creation (mirroring the GPU config and adding OpSlot guards), training config adaptation, test creation, E2E smoke training, and accuracy/performance report generation. Use Gemma 3 as the reference implementation. Trigger: 'NPU migration', 'port model to NPU', 'Ascend support', 'add NPU patch'."
---

# VeOmni GPU→NPU Model Migration Skill

Migrate a model that already has a working VeOmni GPU path
(`veomni/models/transformers/<model>/<model>_gpu_patch_gen_config.py`) to
Ascend NPU. The output is a fully functional NPU patchgen path: NPU
patch config, generated modeling file, training config, E2E script,
tests, and a practice report.

**References (read first, load on demand):**

- `docs/design/patchgen.md` — patchgen DSL, CLI, CI drift check
- `docs/design/kernel_selection.md` — unified kernel registry, OpSlot dispatch
- `docs/design/unified_kernel_registry.md` — per-op backend availability matrix
- `.agents/knowledge/constraints.md` — hard constraints (never edit `generated/`, transformers v5.9.0, FSDP2)
- `.agents/skills/veomni-new-model/SKILL.md` — new-model lifecycle (GPU path only)
- `.agents/skills/veomni-migrate-transformers-v5/SKILL.md` — patchgen DSL reference

**Reference implementations (study before starting):**

- `veomni/models/transformers/qwen3/` — OpSlot-guard pattern (GPU config has
  OpSlot guards; NPU config mirrors GPU functions and re-registers them)
- `veomni/models/transformers/seed_oss/` — direct-NPU-kernel pattern (NPU
  config calls NPU kernels directly instead of going through OpSlot)
- `veomni/models/transformers/gemma3/` — **the migration worked-example**
  produced by this Skill

---

## Phase 1: Pre-Migration Analysis

### 1.1 Confirm GPU path exists

```bash
ls veomni/models/transformers/<model>/
# Must contain:
#   __init__.py
#   <model>_gpu_patch_gen_config.py
#   generated/patched_modeling_<model>_gpu.py
```

If the GPU path does not exist, use the `veomni-new-model` skill first.

### 1.2 Scan for CUDA/GPU-specific dependencies

Search the GPU patch config, the generated GPU modeling file, and the
upstream HF modeling source for device-specific code:

```bash
# In the GPU patch config:
rg "flash_attention\|flex_attention\|triton\|liger\|cuda\|\.cuda()\|is_cuda" \
   veomni/models/transformers/<model>/<model>_gpu_patch_gen_config.py

# In the generated GPU modeling:
rg "flash_attention\|flex_attention\|triton\|liger\|cuda\|\.cuda()\|is_cuda" \
   veomni/models/transformers/<model>/generated/patched_modeling_<model>_gpu.py
```

Document every finding in the migration report's "Dependency Scan" section.

### 1.3 Classify each op into an NPU backend category

For each fused op the GPU path uses, determine the NPU strategy:

| Op (OpSlot name) | GPU backend | NPU backend | NPU kernel file |
|---|---|---|---|
| `rms_norm` | `liger_kernel` | `npu` | `veomni/ops/kernels/rms_norm/npu.py` |
| `rotary_pos_emb` | `liger_kernel` | `npu` | `veomni/ops/kernels/rotary/npu.py` |
| `cross_entropy_loss` | `liger_kernel` | `npu` / `chunk_loss` | `veomni/ops/kernels/cross_entropy/` |
| `swiglu_mlp` | `liger_kernel` | **`eager`** (no NPU backend) | — |
| `moe` | `fused_triton` | `fused_npu` | `veomni/ops/kernels/moe/npu_group_gemm.py` |
| `load_balancing_loss` | `triton` | **`eager`** (no NPU backend) | — |
| `attn_implementation` | `flex_attention` / `flash_attention_2` | `sdpa` / `eager` | — (PyTorch built-in via torch_npu) |

> **Rule**: if no NPU backend exists, the field **must** be set to `"eager"`
> in the NPU training config. The NPU validation tables in
> `veomni/arguments/arguments_types.py` (`_NPU_ALLOWED`,
> `_NPU_REQUIRED`, `_NPU_DEFAULT_FALLBACK`) enforce this at config-parse
> time.

### 1.4 Identify model-specific RMSNorm variants

Some models use non-standard RMSNorm formulations:

| Model family | Weight init | Formula | NPU handling |
|---|---|---|---|
| Qwen3, Llama, most | `ones` | `x * rsqrt(var+eps) * weight` | `standard_rms_norm_forward_npu(x, weight, eps)` |
| Gemma 3 | **zeros** | `x * rsqrt(var+eps) * (1.0 + weight)` | Pass `1.0 + self.weight` to OpSlot |
| Qwen3.5 | `zeros` | `x * rsqrt(var+eps) * (1.0 + weight)` | `qwen3_5_rms_norm_forward_npu(x, 1.0+weight, eps)` |

Check the model's RMSNorm class:
```bash
rg "class.*RMSNorm" -A 15 generated/patched_modeling_<model>_gpu.py
```

Look for:
- `self.weight = nn.Parameter(torch.zeros(dim))` → needs `1.0 + weight`
- `self.weight = nn.Parameter(torch.ones(dim))` → standard
- `self.variance_epsilon` vs `self.eps` → attribute name for epsilon

### 1.5 Check attention implementation compatibility

```bash
rg "attn_implementation\|_attn_implementation\|ALL_ATTENTION_FUNCTIONS\|flex_attention\|BlockMask" \
   generated/patched_modeling_<model>_gpu.py
```

- **flex_attention** (BlockMask): CUDA-only. On NPU, use `sdpa` or `eager`.
  The VeOmni masking-utils wrappers (`create_causal_mask`,
  `create_sliding_window_causal_mask`) return tensor masks for `sdpa`/`eager`
  and `BlockMask` for `flex_attention`, so the same forward code works on both
  backends.
- **flash_attention_2**: CUDA-only. Use `sdpa` on NPU.
- **eager**: hardware-agnostic, works everywhere.
- **model-specific attention** (e.g., DeepSeek sparse): check if an NPU path
  exists; if not, use `eager`.

---

## Phase 2: Create NPU Patchgen Config

### 2.1 Create the config file

Create `veomni/models/transformers/<model>/<model>_npu_patch_gen_config.py`.

**Option A — OpSlot-guard pattern (preferred, like Qwen3):**

The GPU config already has OpSlot guards with `use_non_eager_impl` checks.
The NPU config imports the GPU config's patch functions and re-registers
them with the NPU target file, adding NPU-specific OpSlot declarations:

```python
from veomni.models.transformers.<model>.<model>_gpu_patch_gen_config import (
    # ... import all patch functions from GPU config ...
)
from veomni.models.transformers.<model>.<model>_gpu_patch_gen_config import config as gpu_config
from veomni.patchgen.patch_spec import PatchConfig

config = PatchConfig(
    source_module="transformers.models.<model>.modeling_<model>",
    target_file="patched_modeling_<model>_npu.py",
    description="<Model> with VeOmni NPU fused-operator replacements",
)

# Mirror GPU config's imports, helpers, post-import blocks, dropped names
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)

# Add NPU-specific OpSlot declarations (if GPU config doesn't already have them)
config.add_post_import_block("""
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    """)

# Re-register each patched method/function with the same replacement
config.override_method("<Model>RMSNorm.forward", replacement=<rmsnorm_patched>, description="...")
config.replace_function("apply_rotary_pos_emb", replacement=<rotary_patched>, description="...")
config.override_method("<Model>ForCausalLM.forward", replacement=<forcausallm_patched>, description="...")
```

**Option B — Direct NPU kernel pattern (like SeedOss):**

When the GPU config does NOT use OpSlot guards, the NPU config defines its
own patch functions that call NPU kernels directly:

```python
@config.override_method("<Model>RMSNorm.forward", description="Use NPU fused RMSNorm")
def rmsnorm_forward_npu(self, x):
    from veomni.ops.kernels.rms_norm.npu import rms_norm_forward_npu
    return rms_norm_forward_npu(self, x)
```

> **Choose Option A** when the GPU config already has OpSlot guards (the
> guards fall through to eager code when no NPU kernel is bound, giving
> correct CPU-test behavior). **Choose Option B** when the GPU config has
> no OpSlot guards and you need explicit NPU kernel dispatch.

### 2.2 Handle model-specific RMSNorm

If the model uses the Gemma `(1.0 + weight)` formulation, the OpSlot guard
must pass `1.0 + self.weight` and use `self.eps` (not `self.variance_epsilon`):

```python
@config.override_method("Gemma3RMSNorm.forward", description="...")
def rmsnorm_forward_npu(self, x):
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(x, 1.0 + self.weight, self.eps)  # Gemma offset
    # Original HF code below, unchanged.
    ...
```

### 2.3 Wire `__init__.py` for NPU dispatch

Update `veomni/models/transformers/<model>/__init__.py` to select NPU vs GPU
generated file at import time:

```python
from ....utils.device import IS_NPU_AVAILABLE
from ...loader import MODELING_REGISTRY

@MODELING_REGISTRY.register("<model_type>")
def register_<model>_modeling(architecture: str | None):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_<model>_npu import <Model>ForCausalLM, <Model>Model
    else:
        from .generated.patched_modeling_<model>_gpu import <Model>ForCausalLM, <Model>Model
    ...
```

For **VLM** models (e.g. `gemma3`, `qwen3_vl`) the model class is
`<Model>ForConditionalGeneration` paired with `<Model>Model`. Register a
**second** `model_type` and dispatch both classes:

```python
@MODELING_REGISTRY.register("<model>")          # VLM model_type
def register_<model>_modeling(architecture: str):
    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_<model>_npu import (
            <Model>ForConditionalGeneration, <Model>Model,
        )
    else:
        from .generated.patched_modeling_<model>_gpu import (
            <Model>ForConditionalGeneration, <Model>Model,
        )
    if "ForConditionalGeneration" in architecture:
        return <Model>ForConditionalGeneration
    if "Model" in architecture:
        return <Model>Model
    return <Model>ForConditionalGeneration
```

### 2.4 Generate the NPU modeling file

```bash
patchgen veomni.models.transformers.<model>.<model>_npu_patch_gen_config \
  -o veomni/models/transformers/<model>/generated --diff -v
```

### 2.5 Verify drift gate

```bash
patchgen --check
```

Must exit 0 (no drift between checked-in and freshly generated files).

If the model is a VLM (`ForConditionalGeneration`), add a
`<Model>ForConditionalGeneration.forward` patch that mirrors the GPU override
and threads VeOmni's fused-CE OpSlot through the multimodal forward. Reuse the
GPU patch function when possible.

```python
register_override(
    "<Model>ForConditionalGeneration.forward",
    replacement=<model>_for_conditional_generation_forward_patched,
    description="VLM forward with fused cross-entropy via OpSlot",
)
```

**Vision tower** — most vision encoders (e.g. SiglipVisionModel) work with
`eager`/`sdpa` attention and need **no** NPU-specific patches. If the vision
tower has custom CUDA kernels (e.g. Qwen3-VL's window-attention / RoPE), add
OpSlot guards or direct-NPU-kernel replacements for those methods, mirroring
the text-side patterns. See `qwen3_vl_npu_patch_gen_config.py` for a full
vision-tower example (vision attention, block, RoPE, dummy_forward).

**FLOPS for MFU** — VLM FLOPS must include the vision tower. Add a model-specific
estimator in `veomni/utils/count_flops.py` that sums the text-config FLOPS and
the vision-encoder FLOPS, and register it under the VLM `model_type`.

---

## Phase 3: Create NPU Training Config

Create `configs/text/<model>_npu.yaml` (or `configs/multimodal/...` for VLMs).

Base it on the GPU config, then override `ops_implementation`:

```yaml
model:
  model_path: <same as GPU>
  ops_implementation:
    attn_implementation: sdpa          # or eager; never flex_attention/flash on NPU
    rms_norm_implementation: npu       # if NPU backend exists, else eager
    rotary_pos_emb_implementation: npu # if NPU backend exists, else eager
    swiglu_mlp_implementation: eager   # no NPU backend
    cross_entropy_loss_implementation: npu  # or chunk_loss
```

Other config changes:
- `init_device: meta` — required for FSDP2 (same as GPU)
- `ulysses_size` — set to 1 unless SP has been validated on NPU for this model
- `checkpoint.output_dir` — suffix with `_npu` to distinguish from GPU runs
- `wandb.name` — suffix with `_npu`

If the model has `final_logit_softcapping` (e.g., Gemma 3 4B/12B/27B), the
fused-linear cross-entropy path raises — use
`cross_entropy_loss_implementation: chunk_loss` or `eager` instead.

---

## Phase 4: Create Tests

Create `tests/models/test_<model>_npu.py` with:

1. **Import/structure tests** — verify NPU modeling module exposes expected
   classes and OpSlot declarations
2. **OpSlot guard verification** — inspect patched method source to confirm
   guards use the correct NPU dispatch pattern (e.g., `1.0 + self.weight` for
   Gemma-style RMSNorm)
3. **Forward/backward tests** — build model with NPU ops config (using eager
   fallback on non-NPU hardware), verify loss > 0, gradients flow, loss
   decreases over a few steps
4. **RMSNorm parity test** — verify the OpSlot-guarded RMSNorm (eager
   fallback) matches HF upstream output

Tests run on any backend because OpSlot guards fall through to eager HF code
when no fused kernel is bound (e.g., on CPU in CI).

---

## Phase 5: Create E2E Script

Create `scripts/e2e/<model>_npu_e2e.sh`:

```bash
#!/bin/bash
set -euo pipefail
# ... NPU device detection (ASCEND_RT_VISIBLE_DEVICES / /dev/davinci*) ...
# ... PYTORCH_NPU_ALLOC_CONF, MULTI_STREAM_MEMORY_REUSE env vars ...

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  tasks/train_text.py \
  --config configs/text/<model>_npu.yaml \
  --train.max_steps $MAX_STEPS
```

The E2E script should:
- Auto-detect NPU count
- Set NPU-specific environment variables
- Run a short smoke training (default 20 steps)
- Log output to `<model>_npu_e2e.log`

---

## Phase 6: Write Practice Report

Create `docs/npu_migration/<model>_npu_practice_report.md` using the template
in `docs/npu_migration/report_template.md`. The report must include:

1. **Model overview** — architecture, size, attention type
2. **Dependency scan results** — every CUDA/GPU-specific dependency found
3. **NPU backend selection table** — per-op backend mapping
4. **Files created/modified** — complete file list with descriptions
5. **Test results** — test pass/fail, coverage
6. **E2E results** — loss curve, training stability
7. **Accuracy comparison** — NPU vs GPU loss curves (same model, data, config)
8. **Performance** — MFU calculation, seq length, batch size, hardware spec
9. **Issues encountered** — and how they were resolved

---

## NPU-Specific Pitfalls

1. **`flex_attention` is CUDA-only** — the `BlockMask` type is not available
   on NPU. Use `sdpa` or `eager`. The VeOmni masking-utils wrappers handle
   the mask-type difference automatically.

2. **Gemma-style RMSNorm** — Gemma models initialise `weight` to zeros and
   use `(1.0 + weight)`. The NPU `npu_rms_norm` kernel takes the weight
   directly, so pass `1.0 + self.weight`. The attribute name is `self.eps`
   (not `self.variance_epsilon`).

3. **`final_logit_softcapping`** — Gemma 3 (4B+) uses logit softcapping
   (tanh). The fused-linear cross-entropy path does not support this; use
   `chunk_loss` or `eager` for the loss implementation.

4. **No NPU backend for SwiGLU MLP** — must use `swiglu_mlp_implementation:
   eager`. There is no OpSlot guard needed in the NPU patch for MLP; the HF
   default code runs as-is.

5. **No NPU backend for load-balancing loss** — must use
   `load_balancing_loss_implementation: eager` (Triton kernel is CUDA-only).

6. **Ulysses sequence parallel** — set `ulysses_size: 1` unless SP has been
   validated on NPU for this specific model. The transport helpers
   (all-to-all head/sequence exchange) are backend-agnostic but may not have
   been tested with this model's attention structure on NPU.

7. **Generated files** — never edit files under `generated/` manually. Edit
   the `*_npu_patch_gen_config.py` and re-run `patchgen`.

8. **NPU validation tables** — `veomni/arguments/arguments_types.py` has
   `_NPU_ALLOWED`, `_NPU_REQUIRED`, and `_NPU_DEFAULT_FALLBACK` dicts that
   validate ops at config-parse time. If a new op needs NPU support, it must
   be added to these tables.

9. **VLM `ForConditionalGeneration.forward`** — the multimodal forward is
   **not** patched by the GPU text-only config. The NPU config must add its
   own `ForConditionalGeneration.forward` override that threads the fused-CE
   OpSlot, otherwise the VLM path runs the upstream HF loss (no fused CE).

10. **Vision tower kernels** — SiglipVisionModel works without NPU patches,
    but models with custom CUDA vision kernels (Qwen3-VL) need per-method
    OpSlot guards. Always scan the vision encoder for `flash_attention`,
    `triton`, or `cuda` references.

11. **Logit softcapping** — some VLMs use `final_logit_softcapping` (tanh).
    The fused-linear-CE path does not support it; use `chunk_loss`/`eager`.

---

## Quick Checklist

- [ ] GPU path exists and trains successfully
- [ ] CUDA/GPU-specific dependencies scanned and documented
- [ ] NPU backend selected for each op (or `eager` fallback)
- [ ] Model-specific RMSNorm variant identified (standard vs Gemma offset)
- [ ] Attention backend chosen (`sdpa` or `eager`, never `flex_attention`)
- [ ] `<model>_npu_patch_gen_config.py` created
- [ ] `__init__.py` updated with `IS_NPU_AVAILABLE` dispatch
- [ ] NPU modeling file generated via `patchgen`
- [ ] `patchgen --check` passes (no drift)
- [ ] `configs/text/<model>_npu.yaml` created
- [ ] `tests/models/test_<model>_npu.py` created and passing
- [ ] `scripts/e2e/<model>_npu_e2e.sh` created
- [ ] **VLM only**: second `model_type` + `ForConditionalGeneration` dispatch in `__init__.py`
- [ ] **VLM only**: `ForConditionalGeneration.forward` patch in NPU config
- [ ] **VLM only**: vision tower scanned / patched if needed
- [ ] **VLM only**: FLOPS estimator includes vision tower (MFU correct)
- [ ] Practice report written from template
- [ ] `make quality` passes (ruff check + format)
