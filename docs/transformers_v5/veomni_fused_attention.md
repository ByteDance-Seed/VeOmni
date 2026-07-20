# VeOmni Fused Attention Interface

VeOmni registers sequence-parallel FlashAttention and FlexAttention adapters in
Transformers' `ALL_ATTENTION_FUNCTIONS` registry. Models continue to select an
attention implementation through `config._attn_implementation`; VeOmni's
registered names all enter one model-facing facade and then dispatch to a
backend-specific adapter.

## Configuration

Select FlexAttention through the normal ops configuration:

```yaml
model:
  ops_implementation:
    attn_implementation: flex_attention
```

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

Standalone `sliding_window` is rejected because window semantics must be
encoded in the supplied BlockMask. Dropout and remaining kernel validation are
delegated to the pinned Transformers/PyTorch FlexAttention adapter.

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
