---
name: veomni-gpu-to-npu
description: "Plan and migrate an existing VeOmni GPU model implementation to Ascend NPU. Use when the GPU patchgen path already works and the model needs a migration file inventory, dependency auditing, NPU operator selection, patchgen integration, multimodal wrapper adaptation, or focused model tests. Gemma 3 is the worked example."
---

# VeOmni GPU-to-NPU Model Migration

Produce the smallest maintainable NPU model implementation for a model that already works through VeOmni's GPU path. Keep this skill limited to model code, generated patches, registry dispatch, and focused model validation; do not add training configs, launch scripts, datasets, logs, plots, reports, or checkpoints.

Read these references before changing model code:

- `.agents/knowledge/constraints.md`
- `.agents/skills/veomni-patchgen-model/SKILL.md`
- `.agents/skills/veomni-new-op/SKILL.md`
- `docs/design/patchgen.md`
- `docs/design/kernel_selection.md`
- `veomni/ops/README.md`
- `.agents/knowledge/testing.md`

Use `veomni/models/transformers/qwen3/`, `seed_oss/`, `qwen3_vl/`, and `qwen3_5/` as reference implementations. Gemma 3 under `veomni/models/transformers/gemma3/` is the dense text/VLM worked example.

## 1. Audit The Existing Model Patch

Confirm that the installed Transformers version matches the repository pin, then inspect the GPU patch spec, generated model, upstream Transformers model, model registry, and the live operator registry.

Before editing code, report a migration file inventory and ordered implementation steps. Include only files needed for model adaptation; identify files that can be reused unchanged and explain why any expected patch, test, or registry file is unnecessary.

| File | Purpose | Planned change | Validation |
|---|---|---|---|
| `veomni/models/transformers/<model>/...` | NPU patch spec or model dispatch | Add or update the minimum NPU-specific code | PatchGen, import, and focused model tests |
| `veomni/models/transformers/<model>/generated/...` | Generated NPU model artifact | Regenerate from the patch spec; never edit by hand | Run PatchGen twice and check for no diff |
| `tests/models/...` | Model adaptation coverage | Extend the closest existing test when needed | Run the focused test |

Adjust the rows to the target model. Do not add a file just to match this example; record an explicit no-change decision when an existing implementation already covers it.

```bash
python -c "import transformers; print(transformers.__version__)"
rg -n "transformers==" pyproject.toml
ls veomni/models/transformers/<model>/{__init__.py,*_gpu_patch_gen_config.py}
ls veomni/models/transformers/<model>/generated/patched_modeling_*_gpu.py

rg -n "cuda|triton|flash_attention|flex_attention|liger|is_cuda|\.cuda\(" \
  veomni/models/transformers/<model>

rg -n "VeomniOp|resolve_op_impl|attn_implementation|RMSNorm|apply_rotary|cross_entropy" \
  veomni/models/transformers/<model> \
  veomni/ops/kernels
```

For every patched operator, verify the registered implementations and select the NPU backend according to the exact model semantics rather than the backend name alone.

| Operation | Typical GPU backend | Typical NPU choice |
|---|---|---|
| attention | `flex_attention` / flash attention | `sdpa` or `eager` |
| RMSNorm | `liger_kernel` | `torch_npu` when the variant matches |
| rotary embedding | `liger_kernel` | `torch_npu` |
| causal cross entropy | `liger_kernel` | `chunk_loss` or `eager` |
| SwiGLU | fused GPU backend | registered NPU backend or `eager` |
| MoE | `fused_triton` | `fused_npu` |
| load-balancing loss | `triton` | registered NPU backend or `eager` |

Check device-neutral behavior as part of the audit: allocations must derive their device from inputs or parameters, no `.cuda()` calls may remain, dtype conversions must preserve the model contract, and distributed operations must use VeOmni's current parallel-state APIs.

Model-specific normalization and loss semantics are correctness-critical. Select the registry variant whose tensor contract matches the model: standard RMSNorm uses the `standard` variant, while Gemma 3 initializes weights to zero and uses the `offset` variant to apply `1 + weight`. Do not apply that offset again in model code. Likewise, verify soft-capping, ignored labels, shifted labels, and reduction behavior before replacing the upstream loss.

## 2. Implement The NPU Patch

Prefer an NPU sibling patch spec that reuses the complete device-neutral GPU patch. Add only NPU/VLM behavior that is absent from the GPU path.

```python
from copy import deepcopy

from veomni.models.transformers.<model>.<model>_gpu_patch_gen_config import config as gpu_config


config = deepcopy(gpu_config)
config.target_file = "patched_modeling_<model>_npu.py"
config.description = "<Model> with VeomniOp NPU replacements"
```

Model classes must own instance-local `VeomniOp` handles. Resolve the configured implementation during construction and call the handle unconditionally; the registry's `eager` row supplies the CPU/GPU fallback.

```python
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl


self.veomni_rms_norm = VeomniOp(
    "rms_norm",
    "standard",
    resolve_op_impl("rms_norm_implementation"),
)
hidden_states = self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
```

Inspect `OP_REGISTRY` before adding an implementation. If the required `(op, variant, implementation)` row is missing, follow `veomni-new-op`: implement it under `veomni/ops/kernels`, register it with `register_op`, attach `NpuKernelRequirement()`, and keep model-specific argument adaptation in the PatchGen model. Never import a vendor kernel directly from generated modeling. Do not duplicate a GPU patch body merely to change its target file.

Update `__init__.py` so model classes are selected at import time by device availability.

```python
from ....utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    from .generated.patched_modeling_<model>_npu import <classes>
else:
    from .generated.patched_modeling_<model>_gpu import <classes>
```

Register every supported `model_type` and preserve the existing architecture fallback behavior. A VLM normally requires one registry entry for its multimodal wrapper and another for its text backbone.

## 3. Adapt Multimodal Model Wrappers

Read `.agents/skills/veomni-patchgen-model/references/multimodal.md`, then audit the vision or audio tower, multimodal projector, placeholder masks, packed metadata, sequence-parallel boundaries, and the outer `ForConditionalGeneration.forward` independently from the text backbone.

Verify the checkpoint namespace as well as the operator namespace. Nested generated wrappers can change parameter names even when the upstream class loads correctly, so compare source checkpoint keys with `model.named_parameters()` after constructing the generated wrapper and add an explicit conversion mapping when required.

For the Gemma 3 VLM, legacy keys under `language_model.model.*`, `vision_tower.vision_model.*`, and `multi_modal_projector.*` map to the generated wrapper's nested `model.*` names. Treat unexpected keys or uninitialized vision/projector parameters as hard failures; a tied `lm_head` alias may be the only intentional exception.

Keep the multimodal forward contract intact. Gemma 3 requires `pixel_values` and image-aware `token_type_ids` in addition to the text tensors, and its model-specific position-ID function must remain compatible with VeOmni's packing and sequence-parallel paths.

For distributed training, account for ranks that receive text-only microbatches while peer ranks receive multimodal data. Disable unused trainable auxiliary heads, run a minimal dummy forward through every modality component used by the real media path, and attach the result to the text path with a zero-valued dependency. Verify that real-media and text-only microbatches touch the same trainable modality parameters so DDP/FSDP reductions remain symmetric without changing logits or loss.

When replacing the outer VLM loss, preserve the upstream shifted-label and attention-mask behavior. Do not treat successful text-backbone execution as coverage for the multimodal wrapper.

## 4. Generate And Validate Model Artifacts

Generate model files through patchgen and never edit `generated/` by hand.

```bash
patchgen veomni.models.transformers.<model>.<model>_npu_patch_gen_config \
  -o veomni/models/transformers/<model>/generated --diff -v
git diff --check
```

Run the same patchgen command twice and require the second run to leave the generated `.py` and `.diff` unchanged.

Run focused checks for the patch spec and generated model before hardware training.

```bash
python -m py_compile \
  veomni/models/transformers/<model>/<model>_npu_patch_gen_config.py \
  veomni/models/transformers/<model>/generated/patched_modeling_<model>_npu.py

python -c "import veomni.models.transformers.<model>.generated.patched_modeling_<model>_npu"
make quality
```

Extend an existing model test when practical. Cover generated-module import, local `VeomniOp` selection, eager parity, finite forward/backward outputs, gradients, checkpoint conversion, multimodal input propagation when applicable, and patchgen reproducibility. Verify that constructing another model with a different ops configuration does not retarget handles on the first model.

## Completion Checklist

- [ ] Installed Transformers version matches the repository pin
- [ ] GPU-only model dependencies are classified
- [ ] Device-neutral allocations, dtype conversions, and distributed APIs are verified
- [ ] NPU backends are verified in the live operator registry
- [ ] Every model-side operator is an instance-local `VeomniOp`; no module-global dispatch remains
- [ ] Model-specific normalization and loss semantics are preserved
- [ ] NPU patch spec reuses device-neutral GPU patch bodies
- [ ] Text and multimodal model types are registered as applicable
- [ ] Multimodal checkpoint namespaces and input contracts are verified
- [ ] Real-media and text-only distributed microbatches touch the same trainable modality parameters
- [ ] Generated model and diff are reproducible
- [ ] Import, quality, and focused model tests pass
- [ ] No configs, scripts, datasets, logs, plots, reports, or checkpoints are included
