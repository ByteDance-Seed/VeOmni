# Decision model extension contract

[`DecisionModelSpec`](../../veomni/models/decision/base.py#L39) defines the
architecture-specific pieces used by
[`DecisionModelRuntime`](../../veomni/models/decision/accelerated/model_runtime.py#L11)
and [`DecisionTrainer`](../../veomni/trainer/decision_trainer.py#L28). The
current implementation registers CLEF at import time.

## Model spec

| Member | Contract |
| --- | --- |
| `name` | Registry key selected by `decision.decision_model`. |
| `build_model` | Builds the model from the common model arguments and architecture-owned `decision_model_config`. |
| `build_preprocessor` | Returns a picklable CPU callable that encodes one request as `(features, metadata)`. |
| `build_loss` / `default_loss` | Returns a callable producing `(weighted_numerator, effective_weight)`. |
| `export` | Writes model weights and architecture assets through the shared checkpoint lifecycle. |
| `decode` | Converts candidate logits to JSON-compatible predictions. |

Register specs at module import time:

```python
SPEC = DecisionModelSpec(
    name="example",
    build_model=build_model,
    build_preprocessor=build_preprocessor,
    build_loss=build_loss,
    default_loss="soft_ce",
    export=export_model,
    decode=decode_predictions,
)
DECISION_MODELS[SPEC.name] = SPEC
```

[`DecisionTrainer._build_model_runtime`](../../veomni/trainer/decision_trainer.py#L62)
passes `decision.decision_model_config` into the runtime, which forwards it to
`build_model` without interpreting it.
[`DecisionModelRuntime._build_model_assets`](../../veomni/models/decision/accelerated/model_runtime.py#L41)
builds the preprocessor from the loaded tokenizer and model config.

## Batch and output contract

[`encode_request`](../../veomni/data/decision/collator.py#L58) combines the
architecture preprocessor output with aligned supervision:

```python
{
    "input_ids": ...,       # [tokens]
    "attention_mask": ...,  # [tokens]
    "decision_meta": {...},
    "decision_supervision": {
        "counts": ...,
        "targets": ...,
        "weights": ...,
        "ordinal_mask": ...,
        "keys": ...,
        "question_keys": ...,
    },
}
```

The preprocessor may return `option_ids` in `decision_meta` to define candidate
order. `encode_request` realigns targets to that order. `score` questions set
`ordinal_mask`; model-private type IDs remain in `decision_meta`.

[`DecisionCollator.__call__`](../../veomni/data/decision/collator.py#L27) uses
`MainCollator` for token packing, pads target distributions to the largest
candidate count, and records each request length. Models must return
`ScoringOutput.logits` shaped `[total_questions, padded_options]` in the same
question order. Packed requests must remain independent inside the model.

[`DecisionScoringLoss`](../../veomni/utils/decision_loss.py#L27), metrics, and
[`decode_decisions`](../../veomni/models/decision/base.py#L17) consume only the
shared supervision fields. Single-option, unsupervised, and padding questions
have zero effective weight.

## Weight and export hooks

A model with weights outside the backbone checkpoint may implement
`get_additional_weight_sources()` and return `(path, parameter_prefix)` pairs.
[`DecisionModelRuntime._additional_weight_sources`](../../veomni/models/decision/accelerated/model_runtime.py#L37)
forwards them to the standard weight loader before FSDP2 wrapping.

CLEF uses this hook for `joint_head.safetensors` under the `head.` prefix. Its
adapter also reshapes scalar release parameters to one-element vectors so FSDP2
can shard them, while its checkpoint converter restores the released tensor
shape during loading.

## Required behavior

A decision model implementation must:

1. Register its spec at import time.
2. Keep preprocessing CPU-safe and worker-picklable.
3. Preserve request boundaries when inputs are packed.
4. Return aligned candidate logits and a finite weighted numerator.
5. Export every asset required to rebuild the model.
6. Add parity coverage for serialization, forward/backward, export, and its
   supported distributed path.

The [CLEF parity test](../../tests/models/decision/test_clef_official_parity.py) covers
the model contract against the pinned official source. The
[FSDP2 test](../../tests/models/decision/test_decision_fsdp.py) covers split-head loading
and weighted gradients. The [loader test](../../tests/data/test_decision_dataloader.py)
covers packing, padding, workers, and resume.
