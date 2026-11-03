# Trainer loss reduction hooks

`BaseTrainer` exposes two hooks for objectives that do not normalize by token
count. Subclasses override them together with `postforward` while retaining the
shared accumulation, clipping, optimizer, scheduler, and callback flow.

## Contract

| Method | Contract |
| --- | --- |
| [`prepare_loss_reduction(micro_batches)`](../../veomni/trainer/base.py#L742) | Runs once per optimizer step with all local micro-batches. The default stores globally reduced supervised-token counts. An override prepares its own step denominator. |
| [`prepare_micro_batch_loss(micro_batch)`](../../veomni/trainer/base.py#L751) | Runs before each forward/backward. The default stores the current token counts; an objective without per-micro-batch state may use a no-op. |
| [`postforward(outputs, micro_batch)`](../../veomni/trainer/base.py#L624) | Returns `(loss, loss_dict, aux_metrics)`. `loss` must already have the scale required for backward. |

[`BaseTrainer.train_step`](../../veomni/trainer/base.py#L755) calls them in this
order:

1. Fetch all micro-batches for the optimizer step.
2. Call `prepare_loss_reduction` once.
3. For each micro-batch, call `prepare_micro_batch_loss`, then forward,
   `postforward`, and backward.
4. Clip accumulated gradients and update the optimizer and scheduler once.

A custom objective must define its effective supervision unit, reduce its
denominator over the same ranks that average gradients, reject or handle a zero
denominator, and reset step-local state on every call.

## Example

[`DecisionTrainer.prepare_loss_reduction`](../../veomni/trainer/decision_trainer.py#L151)
sums effective question weights over the complete step and all-reduces the
result. [`DecisionTrainer.postforward`](../../veomni/trainer/decision_trainer.py#L178)
scales each weighted numerator by `dp_size / global_weight`:

```python
class DecisionTrainer(BaseTrainer):
    def prepare_loss_reduction(self, batches):
        self.global_weight = reduce_question_weights(batches)

    def prepare_micro_batch_loss(self, micro_batch):
        pass

    def postforward(self, output, batch):
        numerator, _ = self.loss(output, batch, self.device)
        loss = numerator * (self.model.parallel_state.dp_size / self.global_weight)
        return loss, {"decision_loss": loss}, {"question_weight": self.global_weight}
```

This keeps gradient accumulation equivalent to one weighted full batch.
Loss-reduction hooks do not by themselves add sequence-parallel support; the
model and gradient topology must also support it.

`tests/trainer/test_aux_metrics.py` checks the default token normalization and
the Decision weighted-update path. `tests/models/decision/test_decision_fsdp.py` covers
the distributed FSDP2 path.
