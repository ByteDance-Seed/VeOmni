# QAT, Low-Precision Ops, and Activation Traps

Implementation invariants not duplicated from `docs/usage/low_precision_training.md` and `docs/design/op_selection.md`. Read before modifying `veomni/ops/kernels/{linear,gated_mlp,moe_experts,kv_round}/`, `veomni/ops/qat/`, or `veomni/ops/activation.py`.

## Variant / Row Contracts

- **Variant** = tensor contract or model geometry (`rms_norm/deepseek_v4`, `kv_round/deepseek_v4`); **implementation** = backend (`eager`, `triton`, `qat`, `torchao`, `deepgemm`, `none`). Never encode tile/block geometry in implementation names or make `eager` a variant; pass recipe params (`block_size`, `scale_fmt`, `quant_features`, `rope_features`, `expert_dtype`) as call kwargs.
- `kv_round` models "off" as the `none` row rather than a missing handle.
- Adding or dropping a quantized site changes numerics silently (`tests/models/transformers/deepseek_v4/test_deepseek_v4_qat.py` pins quantized and unquantized sets).

## `QUANTIZING_IMPLS` & Refusal Rules

- Keep every quantizing row's wrapper `dtype` check and its family's `QUANTIZING_IMPLS` set (`linear._module`, `gated_mlp._module`, `moe_experts`) in sync (`tests/ops/base/test_op_selection.py`, `tests/ops/{linear,gated_mlp,moe_experts}/`).
- **Wrapped projections (LoRA)**: `linear._module` and `gated_mlp._module.render` fall back to module `.forward` for non-quantizing rows (incl. GPU/MLU default `liger_kernel` so dense MLP LoRA works) and **raise** for `QUANTIZING_IMPLS`. Never scope that guard to `impl != "eager"`. Modeling's `_moe_lora_op` rejects `moe_experts.QUANTIZING_IMPLS`.
- Do **not** reject `linear_implementation: qat` / `gated_mlp_implementation: qat` alongside `moe_implementation: deepgemm` (shared experts and dense linears are separate GEMMs).
- **Sharded DTensor operands**: fake-quantization rows issue no collectives and require `Replicate` inputs; refuse `Shard` placements instead of quantizing a local piece (test with placement metadata or >=2 ranks).
- **Optional kernel packages**: declare `requires=(...)` and lazy-import inside the entry point; never add to the `gpu` extra; raise `requires unavailable package(s)` instead of falling back to `eager`.

## Activation Handling (`veomni/ops/activation.py`)

- `gated_mlp`: `activation=<config.hidden_act>` is a call arg on every row (`standard` and `geglu`); modeling is a single handle call with no `hidden_act` gate. `activation.py` explicitly serves `{"silu", "swish", "gelu", "gelu_pytorch_tanh"}` (do not derive from `ACT2FN` — e.g. `mish` lacks a bit-exact decomposed backward). Keep `gated_mlp` rows as opaque `wrapper`s (except `liger_kernel` raw pair, which must preserve `SavedState`).
- `moe_experts`: `fused_triton` and `qat` call `apply_activation` / `apply_activation_backward` inside a custom `autograd.Function`. Keep `apply_activation_backward` on **ATen backward ops** (`aten.silu_backward`, `aten.gelu_backward`) — never `torch.func.vjp` (breaks under activation offload `saved_tensors_hooks`), never `autograd.grad` on a fresh `requires_grad_(True)` leaf (breaks Dynamo `fullgraph=True`), and never fp32 closed forms (~1% bf16 drift).
- `fused_quack`/`fused_npu`/`fused_mlu` delegate non-SiLU to `eager` (`merged_experts_act_fn_forward`, which owns EP-local indexing for both merged and split weights); `deepgemm` and `qat` **refuse** unsupported activations.

## `torch.compile` Single-Tensor Inference

- `veomni/ops/registry.py` bypasses the generated `autograd.Function` when no input needs grad because Dynamo fails to inline variadic `forward` + trailing keyword attrs for **1-tensor** rows (e.g. `kv_round`; pinned by `tests/ops/base/test_op_entry.py`).
