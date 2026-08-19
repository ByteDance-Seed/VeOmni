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
       -> one of:
            flash_attention_forward(...)
            flex_attention_forward(...)
            magi_attention_forward(...)
```

The facade resolves only VeOmni's private dispatch table; it does not look the
name up in `ALL_ATTENTION_FUNCTIONS` again. This avoids recursive dispatch and
keeps the Flash, Flex, and Magi adapters independently testable.

The backend compute functions are replaceable module-level slots:

- `attention.flash._flash_attention_forward`, defaulting to Transformers'
  `_flash_attention_forward`;
- `attention.flex._flex_attention_forward`, defaulting to Transformers'
  `flex_attention_forward`;
- `attention.magi._magi_attention_forward`, defaulting to VeOmni's architecture-aware FA4 adapter.

The Magi default prepares an explicit `FA4AttnArg` and reuses it while the range tensors and attention shape remain unchanged, avoiding the upstream facade's repeated GPU-to-CPU range conversion in every transformer layer. VeOmni's FA4 autograd function passes that prepared argument directly to MagiAttention's lower-level `fa4_fwd` and `fa4_bwd` functions. SM90 uses the precompiled CUTLASS `ffa_fa3` backend, while SM100 and newer GPUs use the CUTE DSL/JIT backend. VeOmni prepares and validates the selected backend once per device.

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

Query and key ranges are paired half-open token intervals. When `attn_type_map` is present, values mean `0=full`, `1=causal`, `2=inverse causal`, and `3=bidirectional causal`; `None` means full attention for every range. The mask constructor validates tensor structure and static range/type values once. The caller or model-specific mask builder must also ensure that every range endpoint is within the actual post-SP query/key sequence lengths. The default backend converts the tensor metadata into a prepared `FA4AttnArg` and reuses it across layers while the mask and attention shape remain unchanged. The generic adapter does not infer ranges from dense masks or convert a FlexAttention `BlockMask`.

The current adapter requires `cp_size == 1`, batch size 1, zero attention dropout, and NVIDIA SM90 or newer. It accepts SP1 or VeOmni Ulysses sequence parallelism, passes `scaling` as Magi's `softmax_scale`, and passes `softcap`. SM80 and older GPUs are unsupported.

### Unified MagiAttention mask builder

VeOmni registers `create_magi_mask` as the Transformers mask builder for `veomni_magi_attention_with_sp`. Canonical unpacked causal and bidirectional models that call the Transformers mask registry without a 2D attention mask can therefore select MagiAttention without defining another mask builder. Models with richer visibility call the same builder directly with one of these metadata forms:

- `cu_seq_lens_q` and `cu_seq_lens_k` for packed causal or bidirectional sequences;
- explicit `q_ranges`, `k_ranges`, and `attn_type_map` for mixed or asymmetric visibility;
- `q_length` and `kv_length` for one unpacked sequence.

The builder deliberately does not materialize or reverse-engineer an arbitrary Transformers `mask_function`. Predicate-to-range conversion would require an O(sequence length squared) dense mask and cannot preserve every model-specific visibility rule efficiently. A 2D attention mask also does not expose packed boundaries because VeOmni uses an all-ones mask and records boundaries in `position_ids` and precomputed cumulative sequence lengths. Registry calls with a 2D mask but without explicit range metadata are rejected rather than silently allowing cross-sample attention. Models with packed, sliding-window, prefix, multimodal, or mixed visibility must pass declarative metadata explicitly.

The `gpu` extra installs MagiAttention and the CUTE DSL/JIT dependencies used on SM100 and newer GPUs:

```bash
uv sync --extra gpu --dev
```

### Installing the SM90 CUTLASS overlay

SM90 additionally requires a precompiled CUTLASS overlay. Install the verified default matrix after syncing the GPU environment:

```bash
bash scripts/kernel/install_magi_sm90.sh
```

The default configuration enables BF16 and FP16 inputs, the hdim128 kernel bucket, arbitrary-mask forward and backward for nfunc 1, 3, and 5, `MAX_JOBS=2`, `NVCC_THREADS=4`, and NVCC `--split-compile=32`. The hdim128 bucket accepts input head dimensions up to 128, so models with head dimension 64 do not require the separate hdim64 specialization. Use `--dtype bf16` when FP16 runtime dispatch is unnecessary. FP8, softcap, split KV, paged KV, append KV, local attention, PackGQA, varlen, and cluster kernels are excluded from runtime dispatch because the current Magi adapter does not use them.

Use `--print-config` to inspect the resolved build without checking CUDA or compiling. Selected settings can be overridden from the CLI:

```bash
# Inspect the verified default.
bash scripts/kernel/install_magi_sm90.sh --print-config

# Request additional upstream head-dimension buckets.
bash scripts/kernel/install_magi_sm90.sh --dim 64,128,256

# Customize mask functions, dtype coverage, and compiler concurrency.
bash scripts/kernel/install_magi_sm90.sh \
  --dim 128 \
  --nfunc 1,3 \
  --dtype bf16,fp16 \
  --max-jobs 4 \
  --nvcc-threads 2 \
  --split-compile 16
```

Run the script with `--help` for the complete option list. The pinned upstream build always exposes BF16 and provides no corresponding disable flag, so `--dtype fp16` cannot produce a true FP16-only overlay and is rejected. The `--dtype` option controls runtime dtype exposure, but the upstream nfunc generator may still instantiate disabled dtype and feature combinations during compilation. Non-default matrices are forwarded to the pinned upstream build without claiming that they are supported. Dedicated hdim64 or hdim256 arbitrary kernels can fail CUDA 13 compilation with a PTX register-allocation error. In particular, `--dim 64,128,256` is a valid request but is not a verified configuration and does not fall back automatically if compilation fails.

The overlay is intentionally installed after `uv sync`. A later exact `uv sync` can remove it, so rerun the installer before using MagiAttention on SM90.

Standalone `sliding_window` metadata is rejected because all visibility must already be encoded by the range mask. VeOmni's `_MagiFA4Function` passes the prepared argument to MagiAttention's `fa4_fwd` and reuses the same argument for `fa4_bwd`.

With Ulysses, the ranges describe the full sequence after the
sequence-gather/head-scatter exchange and must be identical on every Ulysses
rank. A layer that passes `skip_ulysses=True` must build local ranges by passing
the same flag to `create_magi_mask`. The forward adapter validates range
endpoints against the actual post-exchange query and key lengths before
launching the kernel. A future Magi Context Parallel implementation may reuse
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
active Ulysses group. Ring CP rejects `skip_ulysses` because a partial Ulysses
bypass would desynchronize the unified U×CP layout. The FSDP zero-valued ViT
dummy may skip both Ulysses and CP only while a private scoped sentinel is
active; that sentinel is entered by `dummy_forward` and restores on exit.
Public `skip_sequence_parallel` arguments and the `skip_sequence_parallel`
vision `**kwargs` flag are rejected.
FlexAttention fail-closes inside the private scope because a global BlockMask
is layout-unsafe after skipping Ulysses. Real multimodal or non-causal vision
CP remains fail-closed.

## Scope

This interface consumes model-provided masks and transports attention tensors.
It does not define model-specific masking, data preprocessing, trainer
scheduling, or FSDP policy.
