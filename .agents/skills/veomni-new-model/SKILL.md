---
name: veomni-new-model
description: "Integrate an external model into VeOmni, author or update standalone `veomni/models/transformers/<model>/modeling_<model>.py` modules, convert pretrained weights, or migrate an existing VeOmni GPU model to Ascend NPU. Trigger: 'add model', 'support new model', 'migrate model', 'update modeling_<m>.py', 'migrate model to Ascend NPU'."
---

# Model Integration & Migration

Bring a model into VeOmni (or update / NPU-migrate an existing `modeling_<m>.py`) while preserving architecture and numerical semantics. Default scope for a new model: pretrained weight loading, real-data training run, and fresh-process resume (narrow when the user asks for inference-only, config-only, or NPU-only work).

## Conditional References (Load Only When Needed)

| Situation | Read |
|---|---|
| Source checkpoint needs key/layout conversion | `references/weight-conversion.md` |
| Migrating an existing VeOmni GPU model to Ascend NPU | `references/npu-migration.md` |
| DiT / video diffusion model | `docs/usage/support_new_models/dit_model_guide.md` |
| VLM / Omni multimodal metadata hooks | `.agents/knowledge/multimodal_metadata.md` |
| Wrapping into SeedOmni modules/graphs | `/veomni-seed-omni` (after base model exists) |

## Phase 1: Inspect Source & Set Scope

1. Record source repo/revision, license, VeOmni base SHA, and weight revision/checksum. Keep scratch/downloaded files outside tracked source (e.g. `.agents_workspace/`).
2. Trace config, construction, forward, and the **actual training objective** (an inference pipeline does not define training loss/noise/targets). Classify each component: trainable vs frozen, shapes/layouts/masks/RoPE, checkpoint namespace & tied weights, backend deps.
3. Pick the closest maintained peer (text, VLM, Omni, DiT) and fix the target hardware/topology (default to smallest feasible topology; list missing weights/hardware as explicit blockers rather than silently shrinking scope).

## Phase 2: Modeling, Registration & Ops Binding

1. **Choose the route**:
   - **Transformers**: author a single self-contained `veomni/models/transformers/<m>/modeling_<m>.py` with `# Upstream-Module:` and `# Base-Transformers-Version:` headers (use `# --- Patch.N ---` markers and `[MODIFIED CLASS]` / `[PATCHED FUNCTION]` banners to keep VeOmni customizations reviewable against upstream HF). Never split into `_gpu` / `_npu` files; gate rare structural NPU branches with `IS_NPU_AVAILABLE` (never `not IS_CUDA_AVAILABLE`).
   - **Diffusers / independent DiT**: place under `veomni/models/diffusers/` (reusable HF encoders still follow the Transformers route).
2. **Registration**: register `model_type` at import time in `veomni/models/transformers/<m>/__init__.py` (or diffusers equivalent); verify lookup in a fresh process.
3. **Config & `VeomniOp` ownership** (see `docs/design/module_config.md` & `docs/design/op_selection.md`):
   - Replace HF `PreTrainedModel` with `VeOmniPreTrainedModel as PreTrainedModel` (`veomni/models/modeling_base.py`); custom configs inherit `VeOmniPretrainedConfig` (or `(VeOmniPretrainedConfig, HFConfig)`), calling `super()`.
   - Construct instance-local `VeomniOp` handles from `self.config.ops_config` at `__init__` (never mutable module-level globals or direct `veomni.ops.kernels.<family>.<impl>` imports). Keep input normalization and loss policy in modeling; keep ops tensor-level.
   - Handwritten Diffusers constructors accepting `attn_implementation` must apply non-`None` overrides to config before HF init and handle binding.
4. **Build eager first**: get eager forward/backward working on small inputs before enabling fused kernels or distributed execution. Only add `parallel_plan.py` for ExtraParallel (EP/embedding/PLE) — FSDP2 wrapping is generic.

## Phase 3: Pretrained Weights

- Inspect source keys/shapes/dtypes/tied weights against meta-constructed target parameters.
- Prefer direct loading when compatible; only write a converter (`scripts/model_conversion/<model_name>.py`) when keys/layouts differ — follow `references/weight-conversion.md`. Never conceal mismatches with `strict=False`.

## Phase 4: Data, Config & Trainer Integration

1. Trace a real sample end-to-end (decoding, transforms, collation, conditioning, targets/labels, loss). Reuse existing transforms/collators where semantics match; for VLM/Omni metadata hooks follow `.agents/knowledge/multimodal_metadata.md`.
2. Derive YAML configs from current trainer argument classes and a peer YAML under `configs/{text,multimodal,dit,model_configs}/` — never hard-code machine paths.
3. Reuse `TextTrainer`, `VLMTrainer`, or `DiTTrainer` and their `VeOmniModelRuntime` / `ModelCheckpointManager` / callbacks (subclass runtime hooks only for demonstrated gaps).

## Phase 5: Validation & Resume

Follow `.agents/knowledge/testing.md` and `docs/transformers_v5/testing_new_model.md` in increasing cost order:

1. **Import & static checks**: fresh-process import/registry lookup; for HF-derived `modeling_<m>.py`, run `tests/models/transformers/test_generated_modeling_imports.py` and `tests/models/transformers/test_generated_call_site_signatures.py`.
2. **Unit & numerical parity**:
   - Add a canonical factory in `tests/models/tiny_configs.py`, a case in `tests/models/base/test_auto_registry.py`, and eager fwd/bwd parity under `tests/models/transformers/` (use `make_eager_ops_config()` on CPU; set composite root eager before building HF reference; use config-owned `sdpa` for FP32 accelerator parity when default attention needs half precision).
   - Compare source vs VeOmni with matched weights, inputs, masks, dtype, and identical noise/timesteps for stochastic objectives; declare tolerances before running.
   - Cover one supported accelerator training configuration (fwd + loss + bwd) per claimed backend; put distinct kernel parity under `tests/ops/`.
3. **Real-data training & parallel E2E**:
   - Run a short real-data job verifying finite loss, gradients on expected trainable params, frozen components, and an optimizer update.
   - Extend `tests/e2e/test_e2e_parallel.py`: default 4-rank matrix is dense `(SP1,EP1)/(SP2,EP1)`, ordinary MoE `(SP1,EP1)/(SP2,EP2)`, aligning DP-mean loss and grad norm from matched initial weights/data/batch. For VLM/Omni, include freeze-ViT and asymmetric `dummy_forward` coverage.
4. **Fresh-process resume**: compare uninterrupted `N+1` steps vs `N`-step save + fresh-process 1-step resume (global step, weights, optimizer/scheduler/scaler, RNG, rank-local data cursor, next batch/loss/update; see `docs/usage/checkpoint.md`).
5. **Quality**: run `make quality` and `/veomni-review` before opening the PR. Clearly distinguish passed, failed, blocked, and out-of-scope checks.

## Phase 6: Deliver

Summarize changed files, source revision, exact reproduction commands, validation results, and explicit limitations/blockers; update supported-model/example docs as needed.
