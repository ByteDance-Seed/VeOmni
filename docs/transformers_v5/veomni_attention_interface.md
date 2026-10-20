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

For `sdpa`, `veomni_sdpa`, and `eager`, unmasked (`attention_mask=None`)
batch-size-1 calls carrying multi-segment `cu_seqlens` (`cu_seq_lens_q`/`k` or
`cu_seqlens_q`/`k`) are automatically split per segment inside the wrapper and
concatenated back along the sequence dimension. In `veomni_sdpa`, this split
happens after the Ulysses sequence-gather/head-scatter exchange so global
`cu_seqlens` match the gathered sequence length. Single-segment metadata or
calls already isolated by a 3D/4D dense `attention_mask` strip the packed
metadata without splitting.

## FlexAttention mask contract

The Flex implementation has four responsibilities. These are independent of
Magi's adapter and do not change the model's mask predicate:

| Responsibility | Implementation | Invariant |
| --- | --- | --- |
| Layout and SP | `standard/flex/_inputs.py`, `standard/flex/__init__.py` | Normalize THD/BHSD before exchange, restore the caller's output layout |
| Backend routing | `standard/flex/__init__.py`: `resolve_flex_attention` | Select eligible FLASH calls; retain Triton and explicit backend behavior |
| Variable-length FLASH | `standard/flex/_fa4.py` | Shape-derived Q/K bounds, one dynamic-only compile path, no dLSE |
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

The second result is optional LSE, not attention weights. Flex rejects
`output_attentions=True` before Ulysses communication or backend selection,
whether the backend is automatic, FLASH, or Triton. Use eager attention when
attention weights are required; omitting the option or passing `False` keeps
the existing output contract.

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
Caller-provided `BlockMask` tiles must match these sizes for automatic FLASH
selection; mismatched tiles stay on Triton without changing the mask.
An explicit `kernel_options["BACKEND"]` wins. FLASH does not request LSE
because that backend's backward rejects dLSE. Attention sinks (`s_aux`) need
LSE renormalization, so they stay on Triton unless FLASH was forced, which
is rejected.

### Compiled mask construction

`flex_attention_mask_builder` caches `torch.compile(create_block_mask)` and uses
it by default, including for packed inputs with `cu_seqlens`. This compilation
is separate from FLASH attention execution. It uses PyTorch's automatic dynamic
generalization: an initial length specialization may be followed by a more
general graph when lengths change. A cached callable can therefore own several
graphs; caching the callable alone does not prove graph reuse.

Packing by itself is not a reason to disable compilation. Changing document
boundaries or segment counts changes metadata values, while the packed
predicate consumes per-token tensors. Regression coverage exercises the public
builder with changing lengths, segment layouts, batched flattened coordinates,
and asymmetric Q/K, checks visibility and all sparse tables against references,
and requires repeated inputs to reuse graphs within a limited compile budget.
Arbitrary changes to predicate code or captured Python state can still
specialize; these tests do not promise graph reuse for every custom predicate.

Mask construction composes padding and offset predicates within each call and
passes device, sequence lengths and block sizes directly to `create_block_mask`.
It does not replace Transformers module helpers, so overlapping calls cannot
leave another invocation's device or geometry installed for native HF callers.
The padding composition retains the pinned HF all-valid fast path and right
padding behavior for cached offsets.

Use `compile_block_mask=False` for explicit eager construction. Both paths
set the deprecated `_compile` flag to `False`; the compiled path uses the cached
wrapper instead. Eager construction may materialize
token-square intermediates, so it can cost substantially more memory on long
sequences. Compare compilation cost, steady execution, and peak memory for the
actual workload before selecting it as a fallback.

With pinned PyTorch 2.11, compiled CPU construction with boolean padding and
nonzero offsets can hit an Inductor `VecMask` C++ codegen error. Use
`compile_block_mask=False` for CPU construction if this occurs. CUDA FLASH
validation uses normal GPU codegen; the CPU limitation does not require
disabling compiled mask construction on CUDA.

### Dynamic compilation

FLASH calls always compile a wrapper with `dynamic=True, fullgraph=True`;
callers do not need a compile-mode parameter. Fixed and variable sequence
lengths use the same implementation. Inside the wrapper, a shallow `BlockMask`
takes its sequence lengths from Q/K tensor shapes, preserving all eight sparse
buffers, `BLOCK_SIZE`, and `mask_mod`. Current FLASH lowering supports these
shape-derived symbols, while dynamically captured Python integer lengths can
be rejected.

The adapter requires the original mask bounds to exactly match Q/K lengths
before entering the wrapper. Logical/truncated bounds and special no-op masks
with different bounds are rejected; build a mask for the actual Q/K lengths.
This does not add padding or change visibility. The Transformers Triton path
retains its existing compilation behavior.

`fullgraph=True` makes cache exhaustion raise instead of falling back to eager
dense attention. The dynamic wrapper supports attention within an eager caller.
An outer `torch.compile(dynamic=True)` can change symbol provenance through the
bounds check and remains unsupported; outer static compilation has a dedicated
GPU regression. Additional Python scalar captures in custom mask/score callbacks
are not normalized by this wrapper. Keep per-sample geometry in indexed metadata
tensors; the pinned CuteDSL lowering can also reject zero-dimensional scalar-tensor
captures, so replacing a Python scalar with a scalar tensor is not sufficient.
Dtype, head dimensions and other guards can still produce specializations.

Framework-generated sliding-window, softcap, and Q/K offset callbacks use
indexed device metadata instead of captured Python scalars. Uniform values are
stored once and broadcast as views; callbacks index them with token positions,
because constant-index loads such as `buffer[0]` also fail in the pinned FLASH
code generator. Zero offsets bypass the offset wrapper. Packed coordinates are
precomputed as row-by-token metadata, avoiding symbolic row-width arithmetic
inside the callback. These changes preserve causal/window visibility,
cached suffix alignment, and softcap-before-bias ordering. They do not rewrite
arbitrary user callbacks or change the explicit backend selection.
Packed coordinate mapping preserves the caller's index dtype (Int32 on FLASH)
before invoking nested predicates, so their metadata lookups remain valid.

Keep `mask_mod` / `mask_function` a fixed predicate. Encode per-batch
geometry in captured tensors such as document ids, span ids, or
`cu_seqlens`, and reuse the same callable. Python constants inside the predicate
can specialize static programs or become unsupported captures in dynamic FLASH.
Baking a new split as `q < 256` can retrace or fail on new layouts. Tensor
*values* can change without changing the predicate program.
`flex_attention_mask_builder` compiles `create_block_mask` around
whatever callable the caller passes. It does not rewrite a per-step range
list into a stable predicate.

A packed prefix-plus-causal mask can keep one callable and refresh two
`int32 [S]` buffers each step:

```python
def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    same_document = document_ids[q_idx] == document_ids[kv_idx]
    causal = q_idx >= kv_idx
    same_full_span = (full_span_ids[q_idx] >= 0) & (
        full_span_ids[q_idx] == full_span_ids[kv_idx]
    )
    return same_document & (causal | same_full_span)
```

### FA4 partial-tile compatibility

For pinned FA4 `4.0.0b16`, VeOmni installs a compatibility forwarder before
compilation to protect auxiliary mask metadata on partial tiles. Later sparse
KV blocks can disable sequence bounds checks even when the query tile contains
padded rows. Masking scores after the callback cannot undo an out-of-bounds
metadata read. Causal visibility expressed through a custom Flex callback still
uses this native noncausal/nonlocal callback path.

The forwarder handles native `mask_seqlen=False` calls with a custom mask and
auxiliary tensors. It checks K completeness first for every Q tile, retaining
full native protection for uncertain K indices. On complete K tiles, it adds
Q-only protection only for physical Q padding. The physical Q extent is
`seqlen_q * qhead_per_kvhead_packgqa`, rather than the logical token count alone.
Both native boolean variants remain compile-time constants; runtime tile
selection is uniform across the CTA.

| Tile/call | Action |
| --- | --- |
| Native bounds already enabled, causal/local native path, or no auxiliary callback | Delegate unchanged; native backward keeps its existing protection |
| Complete physical Q and complete K tile | Delegate unchanged after the K-completeness check |
| Padded Q tile and proven complete K tile | Clamp invalid logical Q indices to zero before the callback; AND its result with Q validity |
| Any Q tile with ragged/negative K tile or empty Q | Delegate with full native sequence bounds enabled |

Q-only protection requires `seqlen_q > 0`, `n_block >= 0` and
`(n_block + 1) * tile_n <= seqlen_k`. The incoming false flag alone is not proof
that K is in range: the native sparse index lists are not validated for ordering.
The fallback preserves full protection for those boundary cases. Native code
already converts PackGQA physical rows to logical query/head indices before the
callback; the Q-only callback must not divide the query index again. Its
register-local temporary does not allocate a PyTorch tensor. Invalid Q rows
remain masked to negative infinity, and valid-token visibility, mask buffers
and callable semantics remain unchanged.

The correction does not require sorted sparse indices for K safety. The
ragged-K regression reorders sparse lists for both complete and padded Q tiles,
checking outputs and gradients against a dense visibility oracle.

Although installation runs when VeOmni's Flex adapter is imported, the forwarder
replaces FA4's shared native method. Direct FA4 calls in the same process also
receive the correction when they use the affected auxiliary callback path.
Ordinary VeOmni `flash_attention_4` calls do not supply `mask_mod` or
`aux_tensors`; their native causal/window/varlen paths do not trigger this defect.
Standalone FA4 programs that do not import VeOmni do not receive this local fix.

Corrected forward/backward kernels use a dedicated FA4 JIT-cache
namespace because FA4's package-source fingerprint excludes the repository
forwarder. The forwarder uses `veomni_sm90_aux_bounds_q_only_k_bounds_materialized_fwd` and
`veomni_sm90_aux_bounds_q_only_k_bounds_materialized_bwd`. This prevents preexisting in-memory or
persisted kernels from bypassing the correction. Use a new namespace whenever shader semantics
change. Old cache objects remain referenced, and both replacement caches are
prepared before installation is published; initialization failure can be retried.
Restart the process after updating this layer; installation is idempotent, not a
hot-reload upgrade of an already installed forwarder.

Installation requires the exact pinned package version and expected signature,
and safely skips missing optional dependencies. H20 validation covers THD/BHSD
outputs and Q/K/V gradients, variable-length graph reuse, asymmetric Q/K,
PackGQA and combined outer static compilation. Finite
numerics and allocator-enabled memcheck alone cannot establish memory safety;
use `PYTORCH_NO_CUDA_MEMORY_CACHING=1` for this check. The shared `apply_mask`
method also serves SM80, which has not been hardware-tested here; SM100's
separate method is not replaced. Re-evaluate the compatibility layer when
upgrading FA4; skipping another version does not establish that it fixes the
defect.

FA4 b16's native forward cache omits Q/K/V zero-stride broadcast patterns.
The adapter materializes broadcast views after KV expansion and before FLASH.
Regression tests share the production cache across ordinary and singleton-KV
expanded layouts, including an outer compiled caller. Native caches use a new
namespace to exclude persisted kernels compiled before normalization.

The correction adds no bounds buffer, extra kernel launch or synchronization.
Q-only protection avoids unnecessary K index wrapping on complete K tiles.
Boundary checks still add kernel instructions; public adapter tests establish
correctness and memory safety, not end-to-end training throughput.

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

The K-completeness check and broadcast-view materialization preserve the public
adapter's safety contract. Historical Bagel throughput figures do not measure
these later safety refinements; this PR's regressions establish correctness and
memory safety rather than a new end-to-end speedup.
