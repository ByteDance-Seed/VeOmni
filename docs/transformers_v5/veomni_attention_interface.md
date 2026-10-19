# VeOmni Attention Interface

VeOmni registers sequence-parallel FlashAttention, FlexAttention, SDPA,
SageAttention, and MagiAttention FFA adapters in Transformers'
`ALL_ATTENTION_FUNCTIONS`. There is no `fused_attention_forward` facade and
no replaceable module-level `_flash_attention_forward` slots. Patched models
bind `VeomniOp("attention", "standard", config._attn_implementation)` once in `__init__` / `modify_init` and call
`self.veomni_attn` from `forward`.

Public names such as `flash_attention_2` rewrite to `veomni_flash_attention_2`
(not a `_with_sp` suffix). Ulysses wrapping lives inside those registered
adapters.

Models continue to select an attention implementation through
`config._attn_implementation`; modeling code stores that choice in a local
`VeomniOp("attention", "standard", implementation)` handle.

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
value to `veomni_flex_attention`. Flash values are rewritten in the
same way:

| Public value | VeOmni registry name |
|---|---|
| `flash_attention_2` | `veomni_flash_attention_2` |
| `flash_attention_2_hub` | `veomni_flash_attention_2_hub` |
| `flash_attention_3` | `veomni_flash_attention_3` |
| `flash_attention_3_hub` | `veomni_flash_attention_3_hub` |
| `flash_attention_4` | `veomni_flash_attention_4` |
| `flex_attention` | `veomni_flex_attention` |
| `magi_attention` | `veomni_magi_attention` |

The native Transformers `flex_attention` registry entry is left unchanged.
Only the VeOmni-specific name routes through VeOmni's SP-aware facade.

## Dispatch and backend adapters

The model-facing call path is:

```text
model-local VeomniOp("attention", "standard", implementation)
  -> registered attention row created by attention.lookup(implementation)
       -> ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_default)
            -> flash_attention_forward(...)
             | flex_attention_forward(...)
             | magi_attention_forward(...)
             | sage_attention_forward(...)
             | sdpa_attention_forward(...)
```

`apply_veomni_attention_patch()` registers the concrete `veomni_*` forward
functions and matching mask builders in the Transformers registries. Each ops
row then uses the same Transformers interface lookup as the model-facing path.
The `eager` row falls back to the modeling module's local
`eager_attention_forward`, because Transformers does not register eager
attention globally. The short `flex_attention` and `magi_attention` rows resolve
their `veomni_*` interfaces so they share the adapters' device and backend
behavior when selected directly through the ops registry.

The Magi default prepares an explicit `FA4AttnArg` and reuses it while the range tensors and attention shape remain unchanged, avoiding the upstream facade's repeated GPU-to-CPU range conversion in every transformer layer. VeOmni's FA4 autograd function passes that prepared argument directly to MagiAttention's lower-level `fa4_fwd` and `fa4_bwd` functions. MagiAttention is available on NVIDIA SM100 and newer GPUs through CUTE DSL/JIT. VeOmni validates the query device once per device.

All registered adapters use the Transformers attention-forward convention.
Q/K/V inputs use `[batch, heads, sequence, head_dim]`; the returned attention
output uses `[batch, sequence, heads, head_dim]`.

## FlexAttention mask contract

The Flex implementation has four responsibilities. These are independent of
Magi's adapter and do not change the model's mask predicate:

| Responsibility | Implementation | Invariant |
| --- | --- | --- |
| Layout and SP | `standard/flex/_inputs.py`, `standard/flex/__init__.py` | Normalize THD/BHSD before exchange, restore the caller's output layout |
| Backend routing | `standard/flex/__init__.py`: `resolve_flex_attention` | Select eligible FLASH calls; retain Triton and explicit backend behavior |
| Compiled FLASH execution | `standard/flex/_fa4.py` | Fixed-shape compilation, exact Q/K bounds, no dLSE |
| Pinned FA4 compatibility | `standard/flex/_fa4_sm90_compat.py` | Protect auxiliary metadata before reads, with a versioned native JIT cache |

Paths above are relative to `veomni/ops/kernels/attention/`. Adapter contracts
live in `tests/ops/attention/test_flex_attention.py`; compatibility dispatch,
installation and asymmetric Q/K regressions live in
`tests/ops/attention/test_fa4_sm90_compat.py`.

### Input layouts and mask semantics

`flex_attention_forward` requires a native
`torch.nn.attention.flex_attention.BlockMask`. The model owns visibility
semantics and BlockMask construction; the generic op does not convert a dense
mask or construct model-specific visibility rules.

Flex additionally accepts 3D Q/K/V in `[tokens, heads, head_dim]` layout and
returns `[tokens, heads, value_head_dim]`. All three inputs must have the same
ndim. The adapter normalizes 3D inputs to `[1, heads, tokens, head_dim]` using
views before Ulysses and backend selection. It does not infer axis order from
sizes: `[heads, tokens, head_dim]` and 4D `[batch, tokens, heads, head_dim]` are
not accepted layout conventions. Packed 3D inputs are one concatenated stream;
the supplied BlockMask defines document isolation. Q and K lengths can differ,
but K/V heads and lengths, Q/K head dimensions, and batch sizes must match.

Both layouts use the same 4D kernel wrapper. FLASH mask bounds are checked
against actual Q/K lengths after any Ulysses exchange, including already
exchanged inputs supplied with `skip_ulysses=True`. Outputs are restored to the
caller's layout after inverse exchange. When Triton returns LSE, it has shape
`[heads, tokens]` for 3D input and `[batch, heads, sequence]` for 4D input; FLASH
continues to return `None` for LSE.

Transformers models may pass `sliding_window` metadata alongside a native
BlockMask whose predicate already encodes the window. The adapter accepts but
does not use that integer metadata to reconstruct or alter visibility; the
supplied BlockMask remains the sole mask authority. Calls without a native
BlockMask are rejected. Dropout and remaining kernel validation are delegated
to the pinned Transformers/PyTorch FlexAttention adapter.

### Backend selection

`veomni_flex_attention` stays one public NVIDIA GPU op. The adapter selects
PyTorch's FLASH backend (CuteDSL wrapping FA4) on SM90 and newer when FA4 is
installed. SM80 stays on Triton; unsupported FA4 dtypes and head dimensions
also fall back to Triton. The registered mask builder uses 128×128 blocks
through SM90 and 256×128 on SM100+, as required by the SM100 FA4 backend.
Callers supplying their own `BlockMask` must use the matching tile size.
An explicit `kernel_options["BACKEND"]` wins. FLASH does not request LSE
because that backend's backward rejects dLSE. Attention sinks (`s_aux`) need
LSE renormalization, so they stay on Triton unless FLASH was forced, which
is rejected.

### FLASH compilation and mask construction

FLASH execution uses `torch.compile(..., dynamic=False, fullgraph=True)`.
Repeated calls with the same shapes can reuse the compiled callable; different
sequence lengths or callback guards can require new specializations.
`fullgraph=True` makes cache exhaustion raise instead of materializing eager
dense attention. There is no compile-mode parameter or automatic eager fallback.

The original BlockMask sequence bounds must exactly match the post-Ulysses Q/K
lengths. Logical/truncated bounds and special no-op masks with different lengths
are rejected. THD inputs are normalized before this check.

`flex_attention_mask_builder` builds BlockMask eagerly and chooses the tile size
for the target hardware. It forces Transformers' deprecated `_compile` argument
off. Eager construction can materialize large token-square intermediates; use
appropriately sized inputs. Compiled mask construction and variable-length graph
reuse are separate follow-up work.

### FA4 partial-tile compatibility

Pinned FA4 `4.0.0b16` can disable sequence bounds for later sparse KV blocks even
when the current Q tile contains padding. A custom `mask_mod` can then read
auxiliary metadata using an invalid Q index. Filtering scores after the callback
does not protect the preceding memory read. Causal visibility expressed through
a custom callback can use this same native noncausal/nonlocal path.

Before compilation, VeOmni installs a version/signature-gated forwarder for the
shared native `AttentionMask.apply_mask`. When sequence bounds are disabled,
causal/local native masking is off, and a custom mask plus auxiliary tensors are
present, the forwarder delegates with native `mask_seqlen=True`. Native FA4 then
protects both Q and K indices before callback evaluation and masks invalid rows
or columns. All other calls are delegated unchanged, including backward calls
whose bounds flag is already true. This conservative implementation does not
add a Q-only fast path or dispatch based on whether a Q tile is complete.

The full-bounds kernels use separate `veomni_sm90_aux_bounds_full_materialized_fwd` and
`veomni_sm90_aux_bounds_full_materialized_bwd` JIT-cache namespaces. FA4's own source fingerprint
excludes this repository forwarder; changing its semantics requires a different
namespace. Both caches are prepared before installation is published, and the
old cache objects remain referenced so initialization failures are retryable.
Installation is idempotent; restart the process after updating this layer.

The exact package version and method signature are checked; missing optional
dependencies are skipped safely. Skipping another version does not prove that
version fixes the defect. The shared native SM80/SM90 method is replaced; SM100
uses a separate method. Direct FA4 auxiliary-callback calls in the same process
also receive this protection. Ordinary `flash_attention_4` causal/window calls
without custom callbacks do not trigger the added guard.

H20 checks exercise THD/BHSD outputs and Q/K/V gradients, non-aligned and
asymmetric lengths, and native pack2. Memory-safety checks must disable the
PyTorch caching allocator with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. SM80 and SM100
hardware numerical validation remains separate.

FA4 b16's native forward cache also omits Q/K/V stride/broadcast patterns.
The adapter materializes zero-stride Q/K/V views before FLASH, and regression
tests reuse the production cache across dense and expanded-singleton KV layouts.
The forwarder adds no PyTorch bounds buffer or kernel launch, but full native
bounds handling adds instructions. These tests establish correctness and
memory safety, not end-to-end training throughput.

## MagiAttention mask and execution contract

`magi_attention_forward` requires a caller-owned `MagiAttentionMask`:

```python
MagiAttentionMask(
    q_ranges=...,       # int32 [num_ranges, 2]
    k_ranges=...,       # int32 [num_ranges, 2]
    attn_type_map=...,  # optional int32 [num_ranges]
)
```

Query and key ranges are paired half-open token intervals. When `attn_type_map` is present, values mean `0=full`, `1=causal`, `2=inverse causal`, and `3=bidirectional causal`; `None` means full attention for every range. The mask constructor validates tensor structure and static range/type values once. The caller or model-specific mask builder must also ensure that every range endpoint is within the actual post-SP query/key sequence lengths. The default backend converts the tensor metadata into a prepared `FA4AttnArg` and reuses it across layers while the mask and attention shape remain unchanged. The generic adapter does not infer ranges from dense masks or convert a FlexAttention `BlockMask`.

The current adapter requires `cp_size == 1`, batch size 1, zero attention dropout, and NVIDIA SM100 or newer. It accepts SP1 or VeOmni Ulysses sequence parallelism, passes `scaling` as Magi's `softmax_scale`, and passes `softcap`. SM90 and older GPUs are unsupported.

### Unified MagiAttention mask builder

VeOmni registers `magi_attention_mask_builder` as the Transformers mask builder for `veomni_magi_attention`. Canonical unpacked causal and bidirectional models that call the Transformers mask registry without a 2D attention mask can therefore select MagiAttention without defining another mask builder. The registered builder only consumes `q_length`, `kv_length`, HF offsets, `mask_function`, and `device`. It always returns one unpacked range pair `[[0, q_length]]` / `[[0, kv_length]]`. Passing `cu_seq_lens_q`, `cu_seq_lens_k`, `q_ranges`, or `k_ranges` as extra kwargs is ignored and drops packed sample isolation.

Packed and mixed visibility must use the mask constructors, not the HF builder:

- `MagiAttentionMask.from_cu_seqlens` for packed causal or bidirectional sequences
- `MagiAttentionMask.from_ranges` for mixed or asymmetric visibility

The shape helper `packed_causal_mask(..., impl="magi_attention")` already routes through `from_cu_seqlens`.

The builder deliberately does not materialize or reverse-engineer an arbitrary Transformers `mask_function`. Predicate-to-range conversion would require an O(sequence length squared) dense mask and cannot preserve every model-specific visibility rule efficiently. A 2D attention mask also does not expose packed boundaries because VeOmni uses an all-ones mask and records boundaries in `position_ids` and precomputed cumulative sequence lengths. Registry calls with a 2D mask are rejected rather than silently allowing cross-sample attention.

The optional `magi` extra requires `gpu` (`veomni[gpu]`) and installs MagiAttention and the CUTE DSL/JIT dependencies used on SM100 and newer GPUs:

```bash
uv sync --extra gpu --extra magi --dev
```

Standalone `sliding_window` metadata is rejected because all visibility must already be encoded by the range mask. VeOmni's `_MagiFA4Function` passes the prepared argument to MagiAttention's `fa4_fwd` and reuses the same argument for `fa4_bwd`.

With Ulysses, the ranges describe the full sequence after the
sequence-gather/head-scatter exchange and must be identical on every Ulysses
rank. A layer that passes `skip_ulysses=True` must build local ranges by passing
the same flag to `create_magi_mask`. The forward adapter validates range
endpoints against the actual post-exchange query and key lengths, and caches
that check with the FA4 metadata while the mask tensors and Q/K shape stay
unchanged. On CUDA this skips later `.all()` reductions; it does not remove a
host sync. A future Magi Context Parallel implementation may reuse
this mask carrier, but distributed dispatch/calc/undispatch and `cp_size > 1`
are outside the current contract.

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
   the generic VeOmni FlexAttention adapter does not recreate them. Prefer a
   fixed `mask_mod` that gathers per-token metadata over a new Python-constant
   closure on every batch.
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

### FLASH eligibility and broadcast layouts

Automatic FLASH selection retains Triton for unsupported sparse Q/V dimensions,
BlockMasks without full-block tables, and trainable `position_bias`. The general sparse training path on SM90/SM100/SM110
requires Q/K and V dimensions in `{32, 64, 96, 128}`. Larger dimensions may
select incompatible native tile sizes or dedicated dense kernels. Automatic
FLASH selection uses BF16/FP16 and excludes unvalidated architectures. Explicit FLASH requests retain native
validation errors; trainable bias and attention sinks raise an early error.

FA4 b16 forward does not key its compiled kernels by Q/K/V zero-stride broadcast
patterns. The adapter materializes only broadcast Q/K/V views, including singleton
KV expansion, before invoking FLASH. Ordinary non-broadcast tensors retain their
layouts. Separate cache namespaces prevent reuse of kernels predating this fix.
