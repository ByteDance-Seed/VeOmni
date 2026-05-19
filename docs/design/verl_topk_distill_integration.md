# verl Integration: Top-K Forward-KL Distillation via the VeOmni Engine

VeOmni's `chunk_topk_distill_function` lets the verl VeOmni engine compute
the top-k forward-KL distillation loss **without materializing the
`[T, V]` student logits tensor**. The kernel rides the same chunked
fused-linear pattern as `chunk_logprobs_function`, so all of verl's
existing fused-kernel infrastructure (`use_fused_kernels=True` +
remove-padding + Ulysses SP) carries over unchanged.

## VeOmni-side contract (this branch)

The model's `forward` accepts two extra kwargs:

```python
teacher_topk_ids: torch.Tensor          # [B, L, K] int64 (dense)
teacher_topk_log_probs: torch.Tensor    # [B, L, K] fp32 teacher log p
log_prob_min_clamp: float | None = None  # optional clamp (DistillationLossConfig)
```

When both teacher tensors are present and `return_log_probs=True`, the
`ForCausalLMLoss` wrapper short-circuits to `chunk_topk_distill_function`
and the model output (`CausalLMOutputWithLogProbs` and its per-model
subclasses for VLM / MoE / Omni) carries three new optional fields:

```python
distillation_losses: torch.Tensor       # [B, L] fp32  (forward KL on top-k)
student_mass:        torch.Tensor       # [B, L] fp32  (detached metric)
teacher_mass:        torch.Tensor       # [B, L] fp32  (detached metric)
```

Sign/range conventions match verl's `compute_forward_kl_topk`:

- `distillation_losses >= 0`
- `student_mass`, `teacher_mass` ∈ `[0, 1]`, `requires_grad=False`
- All three are zero at `IGNORE_INDEX` positions and at the trailing
  pad slot (same masking as `log_probs` / `entropy`).

The kernel is bitwise-equivalent to verl's reference formula
`(exp(log_p_t) * (log_p_t - log_q_s)).sum(-1)` on the top-k support —
proven by `tests/ops/test_chunk_topk_distill.py` and
`tests/models/test_return_log_probs_e2e.py::
test_return_log_probs_with_topk_distill_populates_three_fields`.

## verl-side changes required

Two small patches. Both files already have the surrounding
infrastructure; the new code mirrors the existing `output.log_probs` /
`output.entropy` handling on the `use_fused_kernels=True` path.

### Patch 1 — thread teacher tensors into `model_inputs`

**File:** `verl/workers/engine/veomni/transformer_impl.py`
**Class:** `VeOmniEngineWithLMHead.prepare_model_inputs`
**Anchor:** the existing `use_fused_kernels` + `use_remove_padding`
block (currently `transformer_impl.py:657-662`).

```diff
         use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
         if use_fused_kernels and use_remove_padding:
             shift_labels = output_args["input_ids_rmpad_rolled"].unsqueeze(0)
             model_inputs["labels"] = input_ids_rmpad
             model_inputs["shift_labels"] = shift_labels
             model_inputs["return_log_probs"] = True
+
+            # Thread top-k distillation teacher tensors through to VeOmni's
+            # ForCausalLMLoss wrapper. VeOmni dispatches to
+            # chunk_topk_distill_function when both tensors are present
+            # alongside return_log_probs=True, surfacing per-token
+            # distillation_losses / student_mass / teacher_mass on the
+            # ModelOutput dataclass.
+            distillation_use_topk = tu.get_non_tensor_data(
+                data=micro_batch, key="distillation_use_topk", default=False
+            )
+            if distillation_use_topk:
+                teacher_logprobs = micro_batch["teacher_logprobs"]
+                teacher_ids = micro_batch["teacher_ids"]
+                if teacher_logprobs.is_nested:
+                    teacher_logprobs = teacher_logprobs.values().unsqueeze(0)
+                    teacher_ids = teacher_ids.values().unsqueeze(0)
+                model_inputs["teacher_topk_log_probs"] = teacher_logprobs
+                model_inputs["teacher_topk_ids"] = teacher_ids
+                # Optional clamp from DistillationLossConfig (forwarded only
+                # if the user configured one).
+                clamp = tu.get_non_tensor_data(data=micro_batch, key="log_prob_min_clamp", default=None)
+                if clamp is not None:
+                    model_inputs["log_prob_min_clamp"] = clamp

         return model_inputs, output_args
```

### Patch 2 — read distillation outputs off `model_output` on the fused arm

**File:** `verl/workers/engine/fsdp/transformer_impl.py`
**Class:** `FSDPEngineWithLMHead.prepare_model_outputs`
(inherited by `VeOmniEngineWithLMHead`).
**Anchor:** the existing `if use_fused_kernels:` branch
(currently `transformer_impl.py:1078-1081`), which today reads only
`log_probs` and `entropy`.

```diff
             if use_fused_kernels:
                 # temperature is singleton
                 log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                 entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
+
+                # When distillation_use_topk=True the VeOmni fused-linear
+                # path computes the top-k forward-KL loss in the same
+                # chunked pass that produces log_probs/entropy — no [T, V]
+                # logits materialized. Read the three tensors off the
+                # ModelOutput, apply the same SP gather + nested-tensor
+                # wrap that the non-fused (logits_processor) branch below
+                # already does for the same field names.
+                if distillation_use_topk:
+                    cu_seqlens = input_ids.offsets()
+                    for k, src in (
+                        ("distillation_losses", output.distillation_losses),
+                        ("student_mass", output.student_mass),
+                        ("teacher_mass", output.teacher_mass),
+                    ):
+                        v = src.squeeze(0)
+                        assert v.shape == log_probs.shape, (
+                            f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
+                        )
+                        if self.use_ulysses_sp:
+                            pad_size = output_args["pad_size"]
+                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
+                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
             else:
                 logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                 ...
                 if distillation_use_topk:
                     outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
                     # ... existing non-fused branch ...
```

## What stays exactly the same

- `verl.trainer.distillation.fsdp.losses.compute_forward_kl_topk` is the
  numerical reference. We don't replace it; on the fused path the kernel
  is what runs instead, and it's bitwise-equivalent to the same formula.
- `verl.trainer.distillation.losses.distillation_ppo_loss`,
  `compute_topk_loss`, the `DistillationConfig` / `DistillationLossConfig`
  dataclasses, the data fields (`teacher_logprobs`, `teacher_ids`,
  `distillation_use_topk`), and the policy-loss combine all stay
  unchanged.
- The non-fused (`use_fused_kernels=False`) path is untouched — the
  existing `logits_processor_func` branch already handles it.
- Estimator-only loss modes (`k1`, `k2`, `k3`, `kl`, `low_var_kl`,
  `abs`, `mse`) are not affected: they consume only the existing
  `log_probs` field, which is still populated identically.

## Why this is "without pain"

1. **Pure additive change**: no existing call sites or fields change
   semantics. Code paths that don't set `distillation_use_topk=True` see
   zero behavior change.
2. **Symmetric with the existing fused log-probs handling**: the patch
   reads `output.distillation_losses` exactly the way today's code
   reads `output.log_probs`. Same SP gather, same nested-tensor wrap,
   same `model_output[k]` write — just three more keys.
3. **Single source of truth for numerics**: the kernel formula is
   copied verbatim from `compute_forward_kl_topk`; CI in VeOmni
   (`tests/ops/test_chunk_topk_distill.py::
   test_matches_verl_compute_forward_kl_topk`) pins the equivalence so
   future drift breaks tests on this side.
4. **No new dataclasses or data-pipeline changes**: the existing
   `teacher_logprobs` / `teacher_ids` micro-batch fields are reused
   as-is; only the dispatch from the engine's `prepare_model_inputs`
   to the model's `forward` kwargs is new.
5. **Tested end-to-end on VeOmni's side**: the toy-config e2e test
   exercises a real `Qwen3` and `Qwen3VL` build, asserts the three
   fields populate, match the kernel-direct computation bitwise, and
   that backward flows gradients into `lm_head.weight`.

## Verification flow when verl applies the patches

```bash
# On the verl side, after applying the two patches above:
pytest verl/tests/workers/test_distillation_topk_symmetry_on_cpu.py
# CPU smoke test already in verl that covers the engine plumbing —
# under use_fused_kernels=False it stays a regression test, under
# use_fused_kernels=True (toggle in the test) it exercises the new arm.

# Wire up a small actor in `recipe/`-style and confirm:
#  - micro_batch["teacher_logprobs"] flows into model_inputs["teacher_topk_log_probs"]
#  - output.distillation_losses populates with shape (1, total_nnz/sp_size)
#  - model_output["distillation_losses"] is a nested tensor matching log_probs layout
#  - distillation_ppo_loss(...) consumes it without changes
```
