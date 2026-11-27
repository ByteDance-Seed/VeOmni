# Loss Token Count Contract (`_loss_tokens`)

SeedOmni modules that compute a token-averaged loss can opt into
`train.loss_reduction="token_mean"` by reporting the denominator used to reduce
that loss under the `_loss_tokens` key. Like
[`Module Extra State`](module_rng_state.md), this is a key protocol rather than
a base class.

- Graph collection: [`OmniModel.forward`](../../../veomni/models/seed_omni/modeling_omni.py)
- Trainer reduction: [`OmniTrainer.forward_backward_step` / `train_step`](../../../veomni/trainer/omni/omni_trainer.py)
- Reference implementations:
  [`TextEncoder`](../../../veomni/models/seed_omni/modules/text_encoder/accelerated/accelerated.py),
  [`JanusVqvae`](../../../veomni/models/seed_omni/modules/janus/vqvae/accelerated/accelerated.py),
  [`BagelFlowConnector`](../../../veomni/models/seed_omni/modules/bagel/flow_connector/accelerated/accelerated.py),
  [`DeepseekV41TextEncoder`](../../../veomni/models/seed_omni/modules/deepseek_v41/text_encoder/accelerated/accelerated.py),
  [`DeepseekV41Llm`](../../../veomni/models/seed_omni/modules/deepseek_v41/llm/accelerated/accelerated.py)

## Why modules report it

Under `train.loss_reduction="sample_mean"` (default), `OmniTrainer` divides
each micro-batch's loss by `num_micro_steps`. When micro-batches differ widely
in supervised token count, short samples receive disproportionately larger
per-token gradients than long samples.

Under `train.loss_reduction="token_mean"`, each micro-batch's backward is
scaled by `_loss_tokens` and accumulated gradients are divided once by the
window's total tokens before `clip_grad_norm()`. Because each module already
knows its exact target masking and parallel reduction group, reporting
`_loss_tokens` from the module keeps `OmniTrainer` agnostic to label layout and
model-specific shifting rules.

## The contract

1. **Count units**: `_loss_tokens` is a scalar (0-D) integer tensor giving the
   number of supervised items/tokens that `_loss` was averaged over for that
   micro-batch (for causal LM heads, the shifted non-`IGNORE_INDEX` target
   count; for latent flow heads, the valid latent-token count).
2. **Matching reduction group (critical)**: `_loss_tokens` **must** be summed
   over the exact same process group that `_loss` was reduced over
   (`fsdp_group` when present, otherwise `sp_group` when SP is enabled, or
   local when unreduced).
   - When `_loss` is reduced via `reduce_sequence_parallel_loss` /
     `ForCausalLMLoss(..., loss_reduction_group=group)`, its backward multiplies
     local gradients by `world_size(group) * local_tokens / group_tokens`.
     Multiplying `_loss` by `group_tokens` cancels the micro-batch denominator
     exactly.
   - Reporting an **unreduced local** count alongside a **group-reduced** loss
     silently produces wrong (`local_tokens^2 / group_tokens`) weighting across
     ranks.
3. **`post_forward` ownership**: The pure-HF `decode` / `forward` method may
   return a local `"loss_tokens"` count, while the accelerated `post_forward`
   (or accelerated `decode`) owns the distributed reduction and writes
   `"_loss_tokens"` into the batch dict alongside `"_loss"`. `OmniModel.forward`
   pops both keys per node into `result["losses"]` and `result["loss_tokens"]`.
4. **Zero-supervision micro-batches**: When a rank has zero valid targets,
   emit `_loss = 0.0` (anchored to the graph so FSDP2 collectives still run)
   and sum `0` into `_loss_tokens` across the reduction group. If every rank in
   the group has zero targets, `_loss_tokens == 0`, so `loss * 0` contributes
   zero gradient without dividing by zero; if every micro-batch in the
   accumulation window sums to zero tokens, `OmniTrainer` skips gradient
   rescaling and logs `0.0`.
5. **Single-loss and multi-loss scope**:
   - Every active loss key in `result["losses"]` must have a matching key in
     `result["loss_tokens"]` (when a node emits a `{loss_name: tensor}` dict
     for `_loss`, it must emit a matching `{loss_name: tokens}` dict for
     `_loss_tokens`, which `OmniModel.forward` flattens under
     `{node.name}.{loss_name}`). Any loss missing `_loss_tokens` raises
     `ValueError` before backward.
   - **Single-loss graphs** scale backward by `_loss_tokens` and divide
     accumulated gradients once by the window's total tokens at step end — no
     pre-count hook is required.
   - **Multi-loss graphs** (e.g. DeepSeek-V4.1's `ce_loss` + `mtp_loss` +
     `indexer_kl_loss`, where each loss has its own token denominator):
     - With `num_micro_steps == 1`, each loss is already reduced over its own
       group tokens and keeps weight `1` (`0` if its token count is zero).
     - With `num_micro_steps > 1`, each loss module must implement
       `count_loss_tokens(method=..., micro_batches=...)` returning the
       group-reduced window total for each of its loss keys so `OmniTrainer`
       can weight each loss `k` independently by
       `micro_tokens[k] / window_tokens[k]` without distorting relative loss
       scales across losses with different token counts.

## Minimal example

```python
import torch
import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.seed_omni.mixins.training_module_mixin import post_forward


class MyAcceleratedHeadMixin:
    def decode(self, hidden_states, shift_labels, **kwargs):
        ps = get_parallel_state()
        reduce_group = (
            ps.fsdp_group
            if ps.fsdp_group is not None
            else (getattr(ps, "sp_group", None) if getattr(ps, "sp_enabled", False) else None)
        )
        # Compute group-reduced token-mean loss over `reduce_group`.
        loss, num_local_tokens = self._compute_loss(hidden_states, shift_labels, reduce_group)
        loss_tokens = num_local_tokens.detach().to(dtype=torch.int64)
        if reduce_group is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_tokens, op=dist.ReduceOp.SUM, group=reduce_group)
        return {"loss": loss, "loss_tokens": loss_tokens}

    @post_forward("decode")
    def decode_post(self, loss=None, loss_tokens=None, **outputs):
        result = {"conversation_list": self._conversation_carrier}
        if loss is not None:
            result["_loss"] = loss
            if loss_tokens is not None:
                result["_loss_tokens"] = loss_tokens
        return result
```
