# Training with Validation

VeOmni supports training-time validation through the `EvaluateCallback`. When enabled, the trainer periodically evaluates the model on a validation dataset and logs metrics (loss, perplexity, accuracy, etc.) to stdout and wandb.

## Configuration

Validation is controlled by three config fields that already exist in `VeOmniArguments`:

| Field | Location | Default | Description |
|-------|----------|---------|-------------|
| `data.eval_path` | `DataArguments` | `None` | Path to validation data. If `None`, validation is skipped. |
| `train.eval_steps` | `TrainingArguments` | `0` | Evaluate every N optimizer steps. `0` disables step-based eval. |
| `train.eval_epochs` | `TrainingArguments` | `1` | Evaluate every N epochs. `0` disables epoch-based eval. |

This PR adds one new field:

| Field | Location | Default | Description |
|-------|----------|---------|-------------|
| `train.validation_metrics` | `TrainingArguments` | `["loss"]` | List of metrics to compute. Options: `"loss"`, `"perplexity"`, `"accuracy"`, `"token_accuracy"`. |

## Quick Start

Add the following to your training YAML config:

```yaml
data:
  eval_path: /path/to/validation/data
  # ... other data config ...

train:
  eval_steps: 100        # Evaluate every 100 steps
  eval_epochs: 0         # Disable epoch-based eval (optional)
  validation_metrics:
    - loss
    - perplexity
    - accuracy
  # ... other train config ...
```

Or evaluate per-epoch only:

```yaml
data:
  eval_path: /path/to/validation/data

train:
  eval_steps: 0          # Disable step-based eval
  eval_epochs: 1         # Evaluate at the end of every epoch
  validation_metrics:
    - loss
    - perplexity
```

## Available Metrics

All metrics are distributed-aware: per-rank partial sums are aggregated via `all_reduce` so the final value reflects the entire validation set across all DP ranks.

| Metric | Registry Key | Description |
|--------|-------------|-------------|
| Validation Loss | `"loss"` | Average cross-entropy loss. Uses the model's native loss when available. |
| Perplexity | `"perplexity"` | Token-weighted perplexity: `exp(total_nll / total_tokens)`. Correctly handles variable-length sequences. |
| Accuracy | `"accuracy"` | Classification accuracy: `argmax(logits) == labels`, averaged over valid (non-ignored) positions. |
| Token Accuracy | `"token_accuracy"` | Same computation as accuracy, useful when both sequence-level and token-level metrics are needed. |

## Adding Custom Metrics

Evaluators use the `Registry` pattern. To add a custom metric:

```python
from veomni.data.evaluator import EVALUATOR_REGISTRY, Evaluator
from typing import Dict
import torch

@EVALUATOR_REGISTRY.register("f1_score")
class F1Evaluator(Evaluator):
    def compute(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Compute per-rank partial sums
        predictions = logits.argmax(dim=-1)
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        return {"f1_tp": tp.float(), "f1_tp_count": tp.float(),
                "f1_fp": fp.float(), "f1_fp_count": fp.float(),
                "f1_fn": fn.float(), "f1_fn_count": fn.float()}

    def aggregate(self, partial):
        # all_reduce and compute final F1
        ...
```

Then use it in your config:

```yaml
train:
  validation_metrics:
    - loss
    - f1_score
```

## How It Works

1. **Trigger**: `EvaluateCallback` checks `eval_steps` / `eval_epochs` at the end of each training step / epoch.
2. **Data Loading**: On first call, `build_validation_dataloader` builds a `DistributedDataloader` from `data.eval_path` using the same `StatefulDistributedSampler` as training, but with `shuffle=False` and `drop_last=False`.
3. **Forward Pass**: The model is switched to `eval()` mode and runs under `torch.no_grad()`. Each validation batch is forwarded through the model.
4. **Metric Computation**: Each evaluator computes per-rank partial sums (e.g., total NLL and token count for perplexity). After all batches, partials are `all_reduce`d across DP ranks to produce global averages.
5. **Logging**: Results are logged via `logger.info_rank0()` and optionally to wandb.

## Notes

- Validation does **not** affect training: gradients are not computed, optimizer is not stepped.
- The model is restored to `train()` mode after validation.
- Validation dataloader is built once and cached for subsequent evaluations.
- All metrics respect the HuggingFace `ignore_index=-100` convention for masked positions.
