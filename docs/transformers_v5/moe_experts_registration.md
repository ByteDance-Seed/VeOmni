# VeOmni MoE Experts Registration (Transformers v5)

## Context

For `transformers>=5.0.0`, many MoE expert modules (including Qwen3-MoE) are decorated with `@use_experts_implementation`.
Instead of replacing the experts class itself, VeOmni registers a custom experts backend and lets HF dispatch to it.

## How registration works

VeOmni registers a custom experts implementation key:

- Key: `veomni_fused_moe`
- Registration point: `veomni/ops/fused_moe/__init__.py`

Relevant VeOmni code:

- `veomni/ops/fused_moe/__init__.py`: `VEOMNI_FUSED_MOE_EXPERTS_IMPL`
- `veomni/ops/fused_moe/__init__.py`: `veomni_fused_moe_experts_forward(...)`
- `veomni/ops/fused_moe/__init__.py`: `_register_veomni_fused_moe_experts_impl()`
- `veomni/ops/fused_moe/__init__.py`: `ALL_EXPERTS_FUNCTIONS.register(...)`

HF dispatch path:

- `transformers/integrations/moe.py`: `ALL_EXPERTS_FUNCTIONS = ExpertsInterface()`
- `transformers/integrations/moe.py`: `@use_experts_implementation` wraps `forward()` and dispatches via `ALL_EXPERTS_FUNCTIONS.get_interface(...)`

Equivalent upstream source path:

- `src/transformers/integrations/moe.py`

## Why we set `_experts_implementation` after model load

HF validates experts implementation strings during model initialization. In current Transformers v5, only these are accepted:

- `eager`
- `grouped_mm`
- `batched_mm`

Validation location:

- `transformers/modeling_utils.py`
  - `PreTrainedModel.get_correct_experts_implementation(...)`

Setter path that also re-validates:

- `transformers/modeling_utils.py`
  - `PreTrainedModel.set_experts_implementation(...)`

Because `veomni_fused_moe` is a custom key, setting it too early (or via HF setter) fails validation.
So VeOmni does:

1. mark intended value in config during argument handling
2. set `model.config._experts_implementation` after `loader.load_model(...)`

Relevant VeOmni flow:

- `veomni/models/auto.py`: save marker `config._veomni_experts_implementation`
- `veomni/models/auto.py`: set `model.config._experts_implementation` post-init

## Post-build verification

After build, VeOmni verifies that HF dispatch resolves to VeOmni fused experts forward for configured model types.

Relevant VeOmni code:

- `veomni/models/auto.py`: `MOE_EXPERTS_DISPATCH_CHECK_MODEL_TYPES`
- `veomni/models/auto.py`: `_get_moe_experts_module_for_dispatch_check(...)`
- `veomni/models/auto.py`: `_verify_fused_moe_experts_dispatch(...)`

Current checked model list:

- `qwen3_moe`

To extend coverage, add model types to `MOE_EXPERTS_DISPATCH_CHECK_MODEL_TYPES` and ensure their experts module layout matches `model.layers[0].mlp.experts`.

## Current limitations

In `veomni_fused_moe_experts_forward(...)`, VeOmni currently requires:

- merged expert weights: `gate_up_proj` + `down_proj`
- `is_transposed == False`
  - assertion message notes that HF currently uses `is_transposed=True` mainly for GPT-OSS
- no expert bias (`has_bias == False`)
- always materializes split expert weights with `contiguous()` before fused kernel call
  - this may introduce extra memory copies and overhead in some cases

### Why `contiguous()` is currently needed

HF v5 merged experts store gate + up in a single tensor:

- `gate_up_proj`: `[num_experts, 2 * intermediate_size, hidden_size]`
- first half along dim-1 is gate projection, second half is up projection

VeOmni splits this merged tensor into two matrices:

- `fc1_1_weight = gate_up_proj[:, :expert_dim, :]`
- `fc1_2_weight = gate_up_proj[:, expert_dim:, :]`

Those split tensors are slice views. With upstream layouts or later transforms, these views are not guaranteed to match the
packed memory pattern expected by the current fused MoE kernels. For correctness/stability, VeOmni materializes:

- `fc1_1_weight = ...contiguous()`
- `fc1_2_weight = ...contiguous()`
- `fc2_weight = down_proj.contiguous()`

Current limitation:

- we pay an extra materialization cost per forward path until kernels are updated to safely consume strided/view tensors.

Relevant implementation checks:

- `veomni/ops/fused_moe/__init__.py`

## What to check when debugging

1. Confirm registration happened:
   - `ALL_EXPERTS_FUNCTIONS` contains `veomni_fused_moe`
2. Confirm model config after build:
   - `model.config._experts_implementation == "veomni_fused_moe"`
3. Confirm dispatch verification passes in `build_foundation_model(...)`.
4. If assertion fails, verify the experts module layout and whether model type is included in `MOE_EXPERTS_DISPATCH_CHECK_MODEL_TYPES`.
