---
name: veomni-gpu-to-npu
description: "Plan and migrate an existing VeOmni GPU model implementation to Ascend NPU. Use when `veomni/models/transformers/<model>/modeling_<model>.py` already works on GPU and the model needs a migration file inventory, dependency auditing, NPU operator selection, explicit NPU adaptation in the shared `modeling_<model>.py`, multimodal wrapper adaptation, or focused model tests. Gemma 3 is the worked example."
---

# VeOmni GPU-to-NPU Model Migration

Produce the smallest maintainable NPU model adaptation for a model that already works through VeOmni's GPU path. Keep this skill limited to model code (`modeling_<model>.py`), registry dispatch, and focused model validation; do not add training configs, launch scripts, datasets, logs, plots, reports, or checkpoints.

Read these references before changing model code:

- `.agents/knowledge/constraints.md`
- `.agents/skills/veomni-new-model/SKILL.md`
- `.agents/skills/veomni-new-op/SKILL.md`
- `.agents/knowledge/multimodal_metadata.md`
- `docs/design/op_selection.md`
- `veomni/ops/README.md`
- `.agents/knowledge/testing.md`

Use `veomni/models/transformers/qwen3/`, `seed_oss/`, `qwen3_vl/`, and `qwen3_5/` as reference implementations. Gemma 3 under `veomni/models/transformers/gemma3/` is the dense text/VLM worked example.

## 1. Audit The Existing Model Implementation

Confirm that the installed Transformers version matches the repository pin, then inspect `veomni/models/transformers/<model>/modeling_<model>.py`, upstream Transformers modeling, the model registry, and the live operator registry.

Before editing code, report a migration file inventory and ordered implementation steps. Include only files needed for model adaptation; identify files that can be reused unchanged and explain why any expected file edit is unnecessary.

| File | Purpose | Planned change | Validation |
|---|---|---|---|
| `veomni/models/transformers/<model>/modeling_<model>.py` | Unified standalone modeling module | Add or update the minimum NPU-specific `VeomniOp` / `IS_NPU_AVAILABLE` branches | `make check-hf-modeling`, import, and focused model tests |
| `veomni/models/transformers/<model>/__init__.py` | Model registration | Register any additional architecture (e.g., VLM wrapper) | Registry lookup test |
| `tests/models/...` | Model adaptation coverage | Extend the closest existing test when needed | Run the focused test |

Adjust the rows to the target model. Do not add separate `_gpu` / `_npu` modeling files; record an explicit no-change decision when `VeomniOp` runtime dispatch already covers NPU (as in 13+ of the 19 HF-derived models).

```bash
python -c "import transformers; print(transformers.__version__)"
rg -n "transformers==" pyproject.toml
ls veomni/models/transformers/<model>/{__init__.py,modeling_*.py}
python scripts/sync_hf_modeling.py diff <model>

rg -n "cuda|triton|flash_attention|flex_attention|liger|is_cuda|\.cuda\(" \
  veomni/models/transformers/<model>

rg -n "VeomniOp|ops_config|attn_implementation|RMSNorm|apply_rotary|cross_entropy" \
  veomni/models/transformers/<model> \
  veomni/ops/kernels
```

For every operator handle, verify the registered implementations and select the NPU backend according to the exact model semantics rather than the backend name alone.

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

## 2. Adapt `modeling_<model>.py` For NPU

Keep a **single** `veomni/models/transformers/<model>/modeling_<model>.py` file across all backends (CUDA, ROCm, Ascend NPU, Cambricon MLU, CPU).

Model classes must own instance-local `VeomniOp` handles. Resolve the configured implementation from `config.ops_config` during construction and call the handle unconditionally; `OP_REGISTRY` resolves the NPU row on Ascend hosts and falls back to `eager` on CPU.

```python
from veomni.ops import VeomniOp


rms_norm_impl = config.ops_config.rms_norm_implementation if config is not None else "eager"
self.veomni_rms_norm = VeomniOp(
    "rms_norm",
    "standard",
    rms_norm_impl,
)
hidden_states = self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)
```

Inspect `OP_REGISTRY` before adding an implementation. If the required `(op, variant, implementation)` row is missing, follow `/veomni-new-op`: implement it under `veomni/ops/kernels`, register it with `register_op`, attach `NpuKernelRequirement()`, and keep model-specific argument adaptation in `modeling_<model>.py`. Never import `torch_npu` at module top level in `modeling_<model>.py`. When an unavoidable structural difference exists on NPU (e.g., Ascend TopKV2 finite-mask workarounds or host `cu_seqlens` placement), guard it explicitly with `IS_NPU_AVAILABLE` or `tensor.device.type == "npu"` — never `not IS_CUDA_AVAILABLE` (which misclassifies ROCm/MLU/CPU).

Update `__init__.py` to import directly from `.modeling_<model>` and register every supported `model_type`. Preserve existing architecture fallback behavior. A VLM normally requires one registry entry for its multimodal wrapper and another for its text backbone.

## 3. Adapt Multimodal Model Wrappers

Read `.agents/knowledge/multimodal_metadata.md`, then audit the vision or audio tower, multimodal projector, placeholder masks, packed metadata, sequence-parallel boundaries, and the outer `ForConditionalGeneration.forward` independently from the text backbone.

Verify the checkpoint namespace as well as the operator namespace. Compare source checkpoint keys with `model.named_parameters()` after constructing the wrapper and add an explicit conversion mapping when required.

For the Gemma 3 VLM, legacy keys under `language_model.model.*`, `vision_tower.vision_model.*`, and `multi_modal_projector.*` map to the wrapper's nested `model.*` names. Treat unexpected keys or uninitialized vision/projector parameters as hard failures; a tied `lm_head` alias may be the only intentional exception.

Keep the multimodal forward contract intact. Gemma 3 requires `pixel_values` and image-aware `token_type_ids` in addition to the text tensors, and its model-specific position-ID function must remain compatible with VeOmni's packing and sequence-parallel paths.

For distributed training, account for ranks that receive text-only microbatches while peer ranks receive multimodal data. Disable unused trainable auxiliary heads, run a minimal dummy forward through every modality component used by the real media path, and attach the result to the text path with a zero-valued dependency. Verify that real-media and text-only microbatches touch the same trainable modality parameters so DDP/FSDP reductions remain symmetric without changing logits or loss.

When replacing the outer VLM loss, preserve the upstream shifted-label and attention-mask behavior. Do not treat successful text-backbone execution as coverage for the multimodal wrapper.

## 4. Validate Model Artifacts

Run focused checks on `modeling_<model>.py` before hardware training:

```bash
python -m py_compile veomni/models/transformers/<model>/modeling_<model>.py
python scripts/sync_hf_modeling.py check
PYTHONPATH=. pytest --noconftest tests/models/transformers/test_generated_call_site_signatures.py
python -c "import veomni.models.transformers.<model>.modeling_<model>"
make quality
```

Extend an existing model test when practical. Cover module import, local `VeomniOp` selection, eager parity, finite forward/backward outputs, gradients, checkpoint conversion, and multimodal input propagation when applicable. Verify that constructing another model with a different ops configuration does not retarget handles on the first model.

## Completion Checklist

- [ ] Installed Transformers version matches the repository pin (`make check-hf-modeling` passes)
- [ ] GPU-only model dependencies are classified
- [ ] Device-neutral allocations, dtype conversions, and distributed APIs are verified
- [ ] NPU backends are verified in the live operator registry
- [ ] Every model-side operator is an instance-local `VeomniOp`; no module-global dispatch remains
- [ ] Model-specific normalization and loss semantics are preserved
- [ ] Single `modeling_<model>.py` file serves all backends cleanly (`IS_NPU_AVAILABLE` used only where structurally required)
- [ ] Text and multimodal model types are registered as applicable
- [ ] Multimodal checkpoint namespaces and input contracts are verified
- [ ] Real-media and text-only distributed microbatches touch the same trainable modality parameters
- [ ] Import, signature check, quality, and focused model tests pass
- [ ] No configs, scripts, datasets, logs, plots, reports, or checkpoints are included
