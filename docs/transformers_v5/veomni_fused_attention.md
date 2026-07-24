# VeOmni Fused Attention Interface

VeOmni registers sequence-parallel FlashAttention and FlexAttention adapters in
Transformers' `ALL_ATTENTION_FUNCTIONS` registry. Models continue to select an
attention implementation through `config._attn_implementation`; VeOmni's
registered names all enter one model-facing facade and then dispatch to a
backend-specific adapter.

## Configuration

The ops configuration selects FlexAttention only after the model has been
integrated with the Transformers attention and mask registries. Changing this
value alone does not make an arbitrary model FlexAttention-compatible:

```yaml
model:
  ops_implementation:
    attn_implementation: flex_attention
```

The model must call the attention implementation selected by
`config._attn_implementation` and must supply a native `BlockMask` whose
predicate preserves all model-specific visibility rules. A model that
hard-codes SDPA/FlashAttention, constructs only dense masks, or bypasses
Transformers' mask registry needs model-level patchgen adaptation first.

With `MODELING_BACKEND=veomni`, `OpsImplementationConfig` rewrites this public
value to `veomni_flex_attention_with_sp`. Flash values are rewritten in the
same way:

| Public value | VeOmni registry name |
|---|---|
| `flash_attention_2` | `veomni_flash_attention_2_with_sp` |
| `flash_attention_3` | `veomni_flash_attention_3_with_sp` |
| `flash_attention_4` | `veomni_flash_attention_4_with_sp` |
| `flex_attention` | `veomni_flex_attention_with_sp` |

The native Transformers `flex_attention` registry entry is left unchanged.
Only the VeOmni-specific name routes through VeOmni's SP-aware facade.

## Dispatch and backend adapters

The model-facing call path is:

```text
ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  -> fused_attention_forward(...)
       -> flash_attention_forward(...)
       -> flex_attention_forward(...)
```

The facade resolves only VeOmni's private dispatch table; it does not look the
name up in `ALL_ATTENTION_FUNCTIONS` again. This avoids recursive dispatch and
keeps the Flash and Flex adapters independently testable.

The backend compute functions are replaceable module-level slots:

- `attention.flash._flash_attention_forward`, defaulting to Transformers'
  `_flash_attention_forward`;
- `attention.flex._flex_attention_forward`, defaulting to Transformers'
  `flex_attention_forward`.

All three public callables use the Transformers attention-forward convention.
Q/K/V inputs use `[batch, heads, sequence, head_dim]`; the returned attention
output uses `[batch, sequence, heads, head_dim]`.

## FlexAttention mask contract

`flex_attention_forward` requires a native
`torch.nn.attention.flex_attention.BlockMask`. The model owns visibility
semantics and BlockMask construction; the generic op does not convert a dense
mask or construct model-specific visibility rules.

Transformers models may pass `sliding_window` metadata alongside a native
BlockMask whose predicate already encodes the window. The adapter accepts but
does not use that integer metadata to reconstruct or alter visibility; the
supplied BlockMask remains the sole mask authority. Calls without a native
BlockMask are rejected. Dropout and remaining kernel validation are delegated
to the pinned Transformers/PyTorch FlexAttention adapter.

## Integrating a new patchgen model

Before enabling `attn_implementation: flex_attention` for a new model:

1. Inspect the pinned Transformers modeling source. Its attention layer must
   dispatch through `ALL_ATTENTION_FUNCTIONS` using
   `config._attn_implementation`, and its mask preparation must select the
   matching builder from `ALL_MASK_ATTENTION_FUNCTIONS`. Add narrow patchgen
   overrides when either path is hard-coded.
2. Preserve the model's complete visibility contract in a native `BlockMask`.
   Full attention, sliding windows, bidirectional regions, packed-sample
   boundaries, prefix rules, and cache offsets remain model-owned semantics;
   the generic VeOmni FlexAttention adapter does not recreate them.
3. If VeOmni packing or Ulysses changes the mask inputs, replace the relevant
   Transformers mask-helper imports in the patchgen config and pass the
   required metadata through the generated model forward. Packed boundaries
   must be prepared before model forward from full, unsliced sequence metadata;
   do not recompute them inside attention layers after SP slicing. Self-
   attention may use one boundary vector for both query and key visibility,
   while asymmetric attention must forward every Q/K boundary input its mask
   helper requires. Do not edit the generated modeling file directly.
4. Register the generated class in `MODELING_REGISTRY` under the exact config
   `model_type`. If the integration adds a custom config or processor, register
   those in `MODEL_CONFIG_REGISTRY` and `MODEL_PROCESSOR_REGISTRY` as well.
   Import the model package from `veomni.models.transformers` so every
   module-level registration runs at import time.
5. Regenerate with `patchgen ... --diff -v`, review the generated output, run
   `patchgen --check`, and add model-level tests for registry routing, native
   BlockMask type/visibility, forward/backward parity, packing, and Ulysses
   where supported.

Gemma 3 is the concrete reference in this repository. Its patchgen config
replaces the upstream causal/sliding mask-helper imports with VeOmni wrappers
and overrides `Gemma3TextModel.forward` so `cu_seq_lens_q` reaches mask
construction. Gemma 3 uses self-attention, so that one packed-boundary vector
defines both query and key sample membership. The resulting full/sliding
`BlockMask` objects still come from the model's native visibility rules; only
after that adaptation does the `flex_attention` ops setting select the VeOmni
backend.

See [Modeling Code Generation](../design/patchgen.md#adding-a-new-model) for
the complete patchgen generation and drift-check workflow.

## Ulysses sequence parallelism

When Ulysses is active, both backend adapters use the same transport helpers:

1. exchange local-sequence/global-head Q/K/V into
   full-sequence/local-head tensors;
2. execute the selected attention backend;
3. exchange the output back to local-sequence/global-head layout.

The helpers preserve the existing FlashAttention GQA/KV-head repeat behavior.
FlexAttention additionally restores its log-sum-exp tensor and slices a global
one-dimensional `s_aux` tensor to the rank-local query heads.

FlexAttention with Ulysses currently requires a head-broadcast BlockMask
(`BlockMask.shape[1] == 1`). Local head indices restart at zero on every rank;
a head-specific BlockMask would require rank-aware block slicing and global
head-index rebasing. The adapter rejects such a mask instead of silently
applying the wrong head visibility.

Pass `skip_ulysses=True` for a submodule that must execute independently of the
active Ulysses group.

## Scope

This interface consumes model-provided masks and transports attention tensors.
It does not define model-specific masking, data preprocessing, trainer
scheduling, or FSDP policy.
