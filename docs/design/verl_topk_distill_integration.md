# verl Integration: Top-K Forward-KL Distillation via the VeOmni Engine

VeOmni's `chunk_topk_distill_function` computes verl's top-k forward-KL
distillation loss **without materializing the `[T, V]` student logits
tensor** — the kernel rides the same chunked fused-linear pattern as
`chunk_logprobs.py`, so `use_fused_kernels=True` + remove-padding +
Ulysses SP all carry over.

## VeOmni-side contract

Add two kwargs to the model `forward`; the wrapper routes them through
to the kernel:

```python
teacher_topk_ids: torch.Tensor          # [B, L, K] int64 (dense)
teacher_topk_log_probs: torch.Tensor    # [B, L, K] fp32
log_prob_min_clamp: float | None = None # optional, matches DistillationLossConfig
```

When both teacher tensors are present and `return_log_probs=True`, the
output dataclass carries three new optional fields:

| Field | Shape | dtype | Sign / range | Grad |
|-------|-------|-------|--------------|------|
| `distillation_losses` | `[B, L]` | fp32 | `>= 0` | flows |
| `student_mass` | `[B, L]` | fp32 | `[0, 1]` | detached |
| `teacher_mass` | `[B, L]` | fp32 | `[0, 1]` | detached |

All three are `0` at `IGNORE_INDEX` positions and the trailing pad
slot — same masking as `log_probs` / `entropy`. The KL formula is
verbatim verl's: `Σ_k exp(log_p_t,k) · (log_p_t,k - log_q_s,k)` on the
top-k support (with optional clamp on both terms).

## verl-side changes (two patches, ~30 lines total)

Both are pure additive — symmetric with how
`output.log_probs` / `output.entropy` are already handled, no impact
on the non-fused path or any non-distillation flow.

### 1. Thread teacher tensors into `model_inputs`

**`verl/workers/engine/veomni/transformer_impl.py`** — inside the
existing `use_fused_kernels and use_remove_padding` block of
`VeOmniEngineWithLMHead.prepare_model_inputs`:

```python
distillation_use_topk = tu.get_non_tensor_data(
    data=micro_batch, key="distillation_use_topk", default=False
)
if distillation_use_topk:
    teacher_logprobs = micro_batch["teacher_logprobs"]
    teacher_ids = micro_batch["teacher_ids"]
    if teacher_logprobs.is_nested:
        teacher_logprobs = teacher_logprobs.values().unsqueeze(0)
        teacher_ids = teacher_ids.values().unsqueeze(0)
    model_inputs["teacher_topk_log_probs"] = teacher_logprobs
    model_inputs["teacher_topk_ids"] = teacher_ids
    clamp = tu.get_non_tensor_data(data=micro_batch, key="log_prob_min_clamp", default=None)
    if clamp is not None:
        model_inputs["log_prob_min_clamp"] = clamp
```

### 2. Read the three new fields off `model_output`

**`verl/workers/engine/fsdp/transformer_impl.py`** — inside the
existing `if use_fused_kernels:` branch of
`FSDPEngineWithLMHead.prepare_model_outputs` (inherited by
`VeOmniEngineWithLMHead`), right after the existing `log_probs` /
`entropy_rmpad` reads:

```python
if distillation_use_topk:
    cu_seqlens = input_ids.offsets()
    for k, src in (
        ("distillation_losses", output.distillation_losses),
        ("student_mass", output.student_mass),
        ("teacher_mass", output.teacher_mass),
    ):
        v = src.squeeze(0)
        assert v.shape == log_probs.shape
        if self.use_ulysses_sp:
            pad_size = output_args["pad_size"]
            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
```

## What stays unchanged

- `compute_forward_kl_topk`, `distillation_ppo_loss`,
  `compute_topk_loss`, `DistillationConfig` / `DistillationLossConfig`,
  and the `teacher_logprobs` / `teacher_ids` / `distillation_use_topk`
  micro-batch fields — all reused as-is.
- The non-fused `logits_processor_func` branch — untouched.
- Estimator-only loss modes (`k1`, `k2`, `k3`, `kl`, `low_var_kl`,
  `abs`, `mse`) — they consume only `log_probs`, which is still
  populated identically.

## Verification on VeOmni's side

| Test | What it pins |
|------|--------------|
| `tests/ops/test_chunk_topk_distill.py` (10 tests) | Forward / backward numerics vs dense reference (encodes verl's KL formula), chunk-size invariance, IGN masking, clamp consistency between forward & backward, temperature path |
| `tests/models/test_return_log_probs_e2e.py::test_return_log_probs_with_topk_distill_populates_three_fields` (qwen3-text, qwen3_vl-vlm) | The full `model.forward(..., teacher_topk_*=...)` → `outputs.distillation_losses` path on a real toy model. Asserts: fields populated with right shape/range, mass tensors detached, **bitwise-equal to the kernel called directly on `model.lm_head.weight`**, backward reaches `lm_head.weight.grad` |
