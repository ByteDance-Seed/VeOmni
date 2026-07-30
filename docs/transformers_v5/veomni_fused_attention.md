# VeOmni Fused Attention Interface

VeOmni registers sequence-parallel FlashAttention, FlexAttention, and
MagiAttention FFA adapters in Transformers' `ALL_ATTENTION_FUNCTIONS`
registry. Models continue to select an attention implementation through
`config._attn_implementation`; VeOmni's registered names all enter one
model-facing facade and then dispatch to a backend-specific adapter.

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
| `magi_attention` | `veomni_magi_attention_with_sp` |

The native Transformers `flex_attention` registry entry is left unchanged.
Only the VeOmni-specific name routes through VeOmni's SP-aware facade.

## Dispatch and backend adapters

The model-facing call path is:

```text
ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  -> fused_attention_forward(...)
       -> flash_attention_forward(...)
       -> flex_attention_forward(...)
       -> magi_attention_forward(...)
```

The facade resolves only VeOmni's private dispatch table; it does not look the
name up in `ALL_ATTENTION_FUNCTIONS` again. This avoids recursive dispatch and
keeps the Flash, Flex, and Magi adapters independently testable.

The backend compute functions are replaceable module-level slots:

- `attention.flash._flash_attention_forward`, defaulting to Transformers'
  `_flash_attention_forward`;
- `attention.flex._flex_attention_forward`, defaulting to Transformers'
  `flex_attention_forward`;
- `attention.magi._magi_attention_forward`, defaulting to a lazy import of
  MagiAttention's architecture-portable `functional.ffa_fa4_func`.

The Magi default uses the package's existing `FA4AttnFunc` autograd node.
VeOmni adds no custom autograd implementation. On SM80/SM90, its first use on
each device also applies the MagiAttention 1.1.1 tile-helper compatibility
alias and raises the CUDA thread stack limit required by arbitrary-mask
backward. SM100 and newer retain MagiAttention's CUTE DSL/JIT path.

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

## MagiAttention mask and execution contract

`magi_attention_forward` requires a caller-owned `MagiAttentionMask`:

```python
MagiAttentionMask(
    q_ranges=...,       # int32 [num_ranges, 2]
    k_ranges=...,       # int32 [num_ranges, 2]
    attn_type_map=...,  # optional int32 [num_ranges]
)
```

Query and key ranges are paired half-open token intervals. When
`attn_type_map` is present, values mean `0=full`, `1=causal`,
`2=inverse causal`, and `3=bidirectional causal`; `None` means full attention
for every range. The mask constructor validates tensor structure and static
range/type values once. The caller or model-specific mask builder must also
ensure that every range endpoint is within the actual post-SP query/key
sequence lengths. The tensors are then passed directly to MagiAttention's
public non-distributed FFA API. The generic adapter does not infer ranges from
dense masks or convert a FlexAttention `BlockMask`.

The current adapter requires `cp_size == 1`, batch size 1, zero attention
dropout, and NVIDIA SM80 or newer. It accepts SP1 or VeOmni Ulysses sequence
parallelism, passes `scaling` as Magi's `softmax_scale`, and passes `softcap`.
SM100+ uses the CUTE JIT backend installed by the `gpu` extra. SM80/SM90 uses
the separately compiled BF16 CUTLASS backend, specialized for head dimensions
64 and 128 and FFA function counts 1 and 3:

```bash
uv sync --extra gpu --dev
bash scripts/kernel/install_magi_sm80_sm90.sh sm80  # A100 / SM8x
# or: bash scripts/kernel/install_magi_sm80_sm90.sh sm90  # H100 / SM9x
```

Run the architecture-specific installer after the final uv sync, because an
exact later sync removes the overlay package. Without the overlay, VeOmni
imports normally and reports the required command only if `magi_attention` is
selected on SM80/SM90. The precompiled profile does not include FP16; upstream
FFA rejects FP16 inputs with its build-capability guard.

Standalone `sliding_window` metadata is rejected because all visibility must
already be encoded by the range mask. MagiAttention's own FFA autograd
implementation supplies backward; VeOmni does not add a custom autograd node.

With Ulysses, the ranges describe the full sequence after the
sequence-gather/head-scatter exchange and must be identical on every Ulysses
rank. A future Magi Context Parallel implementation may reuse this mask
carrier, but distributed dispatch/calc/undispatch and `cp_size > 1` are outside
the current contract.

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

When Ulysses is active, all three backend adapters use the same transport
helpers:

1. exchange local-sequence/global-head Q/K/V into
   full-sequence/local-head tensors;
2. execute the selected attention backend;
3. exchange the output back to local-sequence/global-head layout.

The helpers preserve the existing FlashAttention GQA/KV-head repeat behavior.
FlexAttention additionally restores its log-sum-exp tensor and slices a global
one-dimensional `s_aux` tensor to the rank-local query heads. MagiAttention
restores the `[sequence, heads]` LSE returned by FFA to
`[batch, heads, local_sequence]`.

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
