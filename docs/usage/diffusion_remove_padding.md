# Diffusion Remove-Padding Interface

`model.use_remove_padding` is a **DiT-only, default-false execution option** for
model-specific packing inside a fixed microbatch. It is independent of
`train.dyn_bsz` and does not change LLM/VLM collators or configuration.

The common interface is implemented by **MiniMax H3** for fixed-target-geometry
FL2VA and visual Ref2VA offline training. See the
[H3 usage and validation notes](../examples/minimax_h3.md#packed-offline-training).
Enabling the option on an unadapted model still raises before condition/model
weight loading. There is no GPU performance claim implied by this interface.

See [RFC #1198](https://github.com/ByteDance-Seed/VeOmni/issues/1198) for design
context. The interface and model consumers are delivered separately. Generic
attention kernels and model-specific layouts are not implemented by this API.

## Configuration and initial boundaries

For a supporting pair, the relevant settings are:

```yaml
model:
  use_remove_padding: true
train:
  training_task: offline_training
  dyn_bsz: false
  micro_batch_size: 2
  global_batch_size: 16
  bsz_warmup_ratio: 0

data:
  dataloader:
    drop_last: true
```

This is an overlay on a supporting DiT recipe, not a complete model/data configuration.
The initial trainer envelope requires:

- FSDP2, with Ulysses, CP, TP, PP and extra-parallel sizes equal to one.
- Offline training, fixed positive integer sample counts (not booleans or floats),
  complete microbatches, and `global_batch_size` divisible by
  `micro_batch_size * dp_size`. An unspecified global size is derived normally.
- No LoRA, parameter/activation offload, or torch.compile until these combinations
  receive model-specific validation.

The public gate checks this envelope before distributed setup. It does **not**
establish GPU support by itself: a consumer must validate its own hardware,
backend, dtype and model-specific input restrictions. Unsupported combinations
must fail explicitly, not silently select a dense path.

When disabled, the current DiT single-sample setup, condition processing, scalar
loss handling and embedding-extraction behavior remain unchanged. A true value
retains the requested fixed microbatch size and derives the corresponding batch
and accumulation settings. It does not schedule batches by token workload.

## Opt-in model and condition pair

The native model declares:

```python
supports_remove_padding = True


def configure_remove_padding(self, *, attn_implementation: str):
    # Validate the resolved backend and configure this newly built instance.
    ...


def forward(self, *, sample_inputs):
    # Validate inputs, pack, execute, restore original sample order.
    ...
```

The condition model declares:

```python
supports_sample_inputs = True


def prepare_samples(self, **collated_inputs):
    # Validate matching input-column lengths and prepare each sample once.
    # Return list[DiffusionSample].
    ...
```

Classes are resolved through the existing loader/registry; there is no second
packing registry. A flag without the required callable is rejected. The model's
configuration hook runs after construction but before the trainer's freeze/LoRA
and distributed-wrap steps. The backend string is the **resolved**
`OpsImplementationConfig.attn_implementation`, which can have a VeOmni-specific
name such as `veomni_flash_attention_2_with_sp`; consumers must validate it, not
assume they receive the original YAML spelling.

External engines can use `validate_diffusion_remove_padding_support` and
`configure_diffusion_remove_padding` from `veomni.models.diffusers.packing` without
constructing `DiTTrainer`. They must enforce their own execution envelope; the
configuration helper does not know their mesh, task or wrapper lifecycle.
It configures a **newly built** model; `enabled=False` is a no-op, not a runtime
unpatch operation. Neither helper changes parameter names or global forwards.

## Prepared input contract

`DiffusionSample` is a `TypedDict`, not a runtime dataclass, so nested tensors
remain visible to FSDP's normal dict/list pytree traversal:

```python
sample = {
    "model_inputs": {...},  # model-ready tensors, including sampled timesteps
    "targets": {...},       # model-specific targets; may be empty
    "metadata": {...},      # positions, shapes, condition counts, etc.
}
```

Noise/timesteps are sampled **before** the packing boundary. Preserve each
sample's original shapes and metadata. The trainer checks the number of prepared
samples and the three dictionaries; the model/condition implementation owns
column-length validation and semantic correctness.

The prepared list is passed through the wrapped root model:

```python
outputs = model(sample_inputs=samples)
```

Never bypass the root by calling `model.dit(...)` or a custom forward method from
the trainer. All model-internal packing and output reconstruction happens inside
that root forward. Model hooks must preserve checkpoint parameter names and
FSDP wrap boundaries.

## Output and loss contract

`DiffusionBatchOutput` contains:

- `sample_predictions`: a list of per-sample dictionaries, in original input
  order, with predictions restored to their original model-specific shapes.
- `sample_losses`: optional named tensors of shape `[B]`. Each entry is one
  sample's already reduced and modality-weighted loss. External RL/inference
  consumers may omit losses; DiT training requires a nonempty loss dictionary.

For equal-sized microbatches of `B` samples and `K` accumulation steps, the trainer
uses `sample_losses[name].mean() / K`. This preserves equal sample weighting
without detaching gradients. Do not average all packed token errors globally:
that gives longer samples more weight. Preserve any existing per-sample modality
and scheduler weights before supplying the `[B]` loss vectors.

The legacy `outputs.loss` contract is used only when the option is disabled.
A scalar packed loss is rejected rather than ambiguously normalized twice.

## Consumer responsibilities and validation

The common interface deliberately does not infer validity from zero tensor values
or position IDs. A model owns valid-row selection, per-attention-domain Q/KV
boundaries, RoPE, timestep/AdaLN mapping, reference handling and output restoration.
Condition tokens and learned registers must not be mistaken for padding.

Every future consumer must verify:

1. Disabled behavior and unchanged parameter/checkpoint names.
2. Identical prepared inputs through unpacked and packed execution: per-sample
   outputs, loss and gradients, including unequal valid lengths and scheduler weights.
3. Sample isolation, original output order and intact gradient checkpointing.
4. Backend backward support and an optimizer-step comparison under each enabled
   distributed/lifecycle combination.
5. Performance against an equivalent ordinary execution baseline, separately from
   correctness. No speedup is implied by declaring the capability.

CPU protocol regressions live in `tests/trainer/test_diffusion_remove_padding.py`
and are listed explicitly in both GPU and NPU unit-test workflows. That suite's
test-only consumer uses an ordinary CPU module. H3 additionally has
`tests/models/test_minimax_h3_remove_padding.py` for native CPU parity and
`tests/trainer/test_minimax_h3_remove_padding_fsdp.py` for two-rank FSDP2
mixed-precision root input/output, gradient-reduction and optimizer-step parity.
The FSDP2 suite is registered in GPU CI; it skips without the required hardware
or local FlashAttention package. Its execution is separate evidence, not something
the generic CPU suite establishes.
