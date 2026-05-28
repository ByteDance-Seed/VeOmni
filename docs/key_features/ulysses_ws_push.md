# Fused Ulysses Pre-attention `ws_push` Kernel

> **Status:** opt-in, GPU-only (Hopper SM 90+). Default remains the
> existing eager Async-Ulysses path — turning the new kernel on requires
> exactly one YAML field.
>
> **Reading order:** This document is a stand-alone integration guide for
> the new `ulysses_qkv_projection_implementation` knob and is intentionally
> **not** spliced into [`ulysses.md`](./ulysses.md) or
> [`design/kernel_selection.md`](../design/kernel_selection.md). Read those
> for the underlying Ulysses SP model and the broader kernel-selection
> framework; this file documents only what's specific to the fused
> `ws_push` backend.

---

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. What `ws_push` does](#2-what-ws_push-does)
- [3. Requirements](#3-requirements)
- [4. Enabling](#4-enabling)
- [5. End-to-end lifecycle](#5-end-to-end-lifecycle)
- [6. Why `OpScope.GLOBAL` with a state manager (not `OpSlot`)](#6-why-opscopeglobal-with-a-state-manager-not-opslot)
- [7. Wrapper-side transparent dispatch](#7-wrapper-side-transparent-dispatch)
- [8. Per-layer `W_qkv_B` cache + per-step invalidation](#8-per-layer-w_qkv_b-cache--per-step-invalidation)
- [9. Backward is unchanged](#9-backward-is-unchanged)
- [10. Checkpoint format compatibility](#10-checkpoint-format-compatibility)
- [11. Limitations & fallbacks](#11-limitations--fallbacks)
- [12. Memory budget & configuration checklist](#12-memory-budget--configuration-checklist)
- [13. Verification & tests](#13-verification--tests)
- [14. Key files](#14-key-files)
- [15. Future work](#15-future-work)

---

## 1. Motivation

The existing **Async Ulysses** path (`accelerator.enable_async=true`,
see [`ulysses.md`](./ulysses.md#-async-ulysses-cp)) decomposes each
attention-prologue into:

```
3 × F.linear(q_weight | k_weight | v_weight) + 3 × all_to_all_tensor(async_op=True)
```

and overlaps GEMM ↔ comm on a single stream. On Hopper we can do
strictly better — fold the three GEMMs and three `all_to_all`s into
**one** CUTLASS/CuTe kernel that:

1. Computes `[bs, local_seq, hidden] @ W_qkv` with a PackQKV epilogue,
2. TMA-stores each `(rank, Q/K/V)` tile **directly into the destination
   peer's NVSHMEM symmetric-memory buffer** in its epilogue,
3. Issues a single system-scope CAS barrier instead of NCCL all-to-all.

No separate communication kernel, no dual-stream choreography. This is
the **`ws_push`** backend of the new `ulysses_qkv_projection_implementation`
knob.

---

## 2. What `ws_push` does

The forward of `AsyncUlyssesQKVProjection` gains an opt-in branch that
takes a long-lived **`WsPushState`** (NVSHMEM symm-mem buffer +
metadata) plus a pre-permuted **`W_qkv_B`** (and optional `bias_B`)
which is the round-robin head-interleave of
`cat([q.weight, k.weight, v.weight], dim=0)`.

```python
q, k, v = ws_push_forward_impl(
    hidden_states, None, ws_push_state,
    W_qkv_B=W_qkv_B, bias_B=bias_B,
)
```

The returned `q / k / v` are zero-copy views into
`ws_push_state.peer_out_buf`. They are immediately `.clone()`'d (so the
next layer's forward can safely overwrite the buffer), unpadded along
the global-sequence dimension, and then handed off to the *unchanged*
post-projection path (QK-norm, etc.) and the *unchanged* hand-written
backward (see [§9](#9-backward-is-unchanged)).

---

## 3. Requirements

| Requirement | Why |
|---|---|
| NVIDIA Hopper (SM 90+), e.g. H100 / H20 | Kernel uses Hopper TMA + `atom.cas.sys` + `fence.acq_rel.sys` |
| PyTorch exposing `torch.distributed._symmetric_memory` | Peer buffers are NVSHMEM-backed |
| `cutlass`, `quack` Python packages | Fused GEMM + PackQKV epilogue is in CuTe DSL on top of `quack.gemm_sm90.GemmSm90` |
| `ulysses_size > 1` and `num_kv_heads % ulysses_size == 0` | Fused kernel cannot expand GQA KV heads at runtime |
| Static `(bs, local_seq, nheads_q/k/v, head_dim, dtype)` for the training run | The symm-mem buffer is sized once at `on_train_begin` and cannot accommodate per-step variations |

Selecting `ws_push` on a non-conforming host raises a clear error at
**config-validation time** (during
`OpsImplementationConfig.__post_init__` → `apply_global_ops`), well
before any training compute starts.

---

## 4. Enabling

That's the entire user-facing change:

```yaml
model:
  ops_implementation:
    ulysses_qkv_projection_implementation: ws_push   # default: "eager"
```

**No code changes are required in the attention modules of Qwen3-VL /
Qwen3-VL-MoE / WAN / your own model.** The wrapper
`async_ulysses_qkv_projection` performs a *transparent dispatch* — see
[§7](#7-wrapper-side-transparent-dispatch).

---

## 5. End-to-end lifecycle

```
YAML  (ulysses_qkv_projection_implementation: ws_push)
  │
  ▼
OpsImplementationConfig.__post_init__              # parse time
  └─ _validate_implementations()
       └─ NPU allow-list rejects 'ws_push' on Ascend (eager-only)
  │
  ▼
build_foundation_model(..., ops_implementation=ops)
  └─ apply_ops_config(ops)
       └─ apply_global_ops(ops)
            sets state_manager._active_impl_name = "ws_push"
            runs _ws_push_side_effect():
              └─ _preflight_ws_push():
                   require CUDA, SM 90+, torch.distributed._symmetric_memory
  │
  ▼
BaseTrainer._init_callbacks()
  └─ self.fused_ulysses_callback = FusedUlyssesStateCallback(self)
        sees impl == 'ws_push'
        pre-extracts (sp_group, device, bs, local_seq,
                      nheads_q/k/v, head_dim, dtype)
  │
  ▼
trainer.train()
  ├─ on_train_begin
  │   └─ FusedUlyssesStateCallback.on_train_begin
  │        WsPushStateManager(...)
  │          calls init_ws_push_state(...)   # collective on sp_group
  │          allocates NVSHMEM symm buffer
  │        set_active_manager(mgr)
  │
  ├─ each attention forward
  │   └─ async_ulysses_qkv_projection(..., ws_push_state=None, W_qkv_B=None, bias_B=None)
  │        if all three are None:
  │           mgr = get_active_manager()
  │           if mgr.is_compatible(bs, local_seq, dtype):
  │               W_qkv_B, bias_B = mgr.get_or_build_for(q_w, k_w, v_w, q_b, k_b, v_b)
  │               ws_push_state  = mgr.state
  │        AsyncUlyssesQKVProjection.apply(..., ws_push_state, W_qkv_B, bias_B)
  │
  ├─ on_step_end
  │   └─ FusedUlyssesStateCallback.on_step_end
  │        mgr.invalidate_weight_cache()       # drop stale per-layer W_qkv_B
  │
  └─ on_train_end
      └─ FusedUlyssesStateCallback.on_train_end
           mgr.teardown()
             torch.cuda.synchronize()
             dist.barrier(sp_group)
             state.close()                     # release symm-mem buffer
           clear_active_manager()
  │
  ▼
trainer.destroy_distributed()
  └─ dist.destroy_process_group()              # only after teardown — order matters
```

---

## 6. Why `OpScope.GLOBAL` with a state manager (not `OpSlot`)

VeOmni's main "configurable kernel" registry (`KERNEL_REGISTRY` +
`OpSlot`) — as documented in
[`design/kernel_selection.md`](../design/kernel_selection.md) — assumes
the resolved kernel is a stateless `Callable[..., Tensor]` bindable
**once at `build_foundation_model` time**.

This kernel cannot fit that model:

- The forward needs an NVSHMEM **symmetric-memory buffer** sized for a
  specific `(bs, local_seq, nheads_*, head_dim, dtype)` combo. The
  buffer is allocated *collectively* across the sequence-parallel group,
  which must happen **after** the process group is up but **before** the
  first forward.
- Each attention layer needs its own pre-permuted `W_qkv_B`
  (`cat([q.weight, k.weight, v.weight])[interleave_perm]`), which must
  be invalidated after every `optimizer.step` because the cat is a value
  snapshot.
- Training-end teardown must run
  `cuda.sync → dist.barrier(sp_group) → state.close()` *before*
  `destroy_process_group`.

So the op follows the same `OpScope.GLOBAL` pattern as
`load_balancing_loss` and `moe`, but with two twists:

```
state_manager._active_impl_name : str
    # bound by apply_global_ops via global_slot (string, not function ptr)

state_manager._active_manager   : WsPushStateManager | None
    # published by FusedUlyssesStateCallback.on_train_begin; cleared by on_train_end
```

The actual dispatch is **not** a function-pointer call — it is a guarded
lookup performed by the transparent wrapper described next.

---

## 7. Wrapper-side transparent dispatch

The wrapper `async_ulysses_qkv_projection` in
`veomni/distributed/sequence_parallel/async_ulysses.py` does an
**opt-in fallback lookup** *only* when the caller did not explicitly
pass any of `ws_push_state / W_qkv_B / bias_B`:

```python
if ws_push_state is None and W_qkv_B is None and bias_B is None:
    try:
        from veomni.ops.kernels.fused_ulysses_projection.state_manager import (
            get_active_manager,
        )
        _mgr = get_active_manager()
    except Exception:
        # NPU / non-symm-mem hosts must never be forced into the fused path.
        _mgr = None

    if _mgr is not None and hidden_states is not None:
        if _mgr.is_compatible(
            bs=hidden_states.shape[0],
            local_seq=hidden_states.shape[1],
            dtype=hidden_states.dtype,
        ):
            W_qkv_B, bias_B = _mgr.get_or_build_for(
                q_weight, k_weight, v_weight, q_bias, k_bias, v_bias,
            )
            ws_push_state = _mgr.state

return AsyncUlyssesQKVProjection.apply(..., ws_push_state, W_qkv_B, bias_B)
```

Three properties fall out of this design:

1. **Existing call sites are zero-modification**. Qwen3-VL / WAN /
   `tests/parallel/ulysses/attention.py` all invoke
   `async_ulysses_qkv_projection(...)` without the new kwargs — the
   wrapper either injects the manager state (when active) or no-ops
   (when not).
2. **Caller override always wins**. If a user explicitly passes
   `ws_push_state=` / `W_qkv_B=` / `bias_B=` the lookup is skipped — the
   caller is in full control. Useful for testing and bespoke fine-tuning
   setups.
3. **Single guarded boundary**. There is no per-forward "which backend?"
   branch deep inside the autograd Function. The branch lives at the
   wrapper boundary only, behind one cheap `if` and one module-level
   global read.

---

## 8. Per-layer `W_qkv_B` cache + per-step invalidation

The fused kernel expects a single `[N_total, hidden]` matrix laid out in
the kernel's round-robin head-interleave order — i.e. the permuted
concatenation of `q.weight | k.weight | v.weight`. Two design choices
follow:

### 8.1 Cache per attention layer, keyed by `id(q_weight)`

Under FSDP2 the **`nn.Parameter` object identity is stable across
forwards**: unshard rebinds the underlying storage, not the Parameter
wrapper itself. So `id(q_weight)` is a safe cache key inside a single
optimizer step. Without the cache, every attention layer would redo a
`cat + permute` on every microbatch forward.

```python
# state_manager.py, simplified
def get_or_build_for(self, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias):
    key = id(q_weight)
    if key in self._cache:
        return self._cache[key]
    with torch.no_grad():
        W_qkv = torch.cat([q_weight, k_weight, v_weight], dim=0)
        W_qkv_B = W_qkv[self._state.interleave_perm, :].contiguous().detach()
        ...
    self._cache[key] = (W_qkv_B, bias_B)
    return self._cache[key]
```

### 8.2 Invalidate at `on_step_end`

The cache holds a **value snapshot** of the weights. After
`optimizer.step` the parameters change but their `id(...)` does not —
without explicit invalidation, the next forward would silently consume
stale weights. So `FusedUlyssesStateCallback.on_step_end` calls
`mgr.invalidate_weight_cache()`, dropping the entire cache; the next
forward rebuilds the snapshot from the post-step weights.

---

## 9. Backward is unchanged

Only the **forward** is fused. `W_qkv_B` is `.detach()`'d and is not
registered as a parameter. Gradients flow through the **unchanged**
backward path of `AsyncUlyssesQKVProjection` (`async_ulysses.py:301-509`)
directly into the original `q.weight / k.weight / v.weight`.

The autograd graph observed by FSDP2 and the optimizer is **identical**
whether `ws_push` is on or off — which is why we can ship the forward
fusion without changing any model code, any patchgen config, or any
optimizer plumbing. The new field
`ulysses_qkv_projection_implementation` is, from the user's perspective,
purely a forward-perf knob.

A fused backward is explicitly out-of-scope for this PR (see
[§15](#15-future-work)).

---

## 10. Checkpoint format compatibility

The `ws_push` path requires `FusedQKVLinear` (a single `nn.Parameter` of
shape `[(n_q + 2·n_kv) · head_dim, hidden]`) on every attention layer.
That changes both the in-memory state_dict keys and the on-disk format —
the model no longer carries `q_proj.weight / k_proj.weight / v_proj.weight`,
it carries `qkv_proj.weight` (and `qkv_proj.bias` when `attention_bias=True`).

VeOmni handles this entirely at the **checkpoint loader** layer; you do
not need to do anything special when launching a fine-tune from a HF
Qwen3-VL / Qwen3-VL-MoE checkpoint.

### 10.1 Loading an HF checkpoint (HF → fused)

A model-level `_create_checkpoint_tensor_converter` factory is registered
via `MODELING_REGISTRY` on every Qwen3-VL / Qwen3-VL-MoE model class.
At load time, the loader (`veomni/models/module_utils.py:335-364`)
instantiates one converter per model and streams every safetensor shard
through it. The converter buffers the three `q/k/v_proj.{weight,bias}`
tensors per attention prefix and emits a single merged
`qkv_proj.{weight,bias}` tensor (via `torch.cat(..., dim=0)`) once all
three arrive. The dispatch is keyed by regex on the full key path —
unlike a module-level `_register_load_state_dict_pre_hook`, the
converter can see the whole `model.language_model.layers.N.self_attn.*`
namespace and rename `q/k/v_proj → qkv_proj` correctly.

The relevant files:

| Model | Converter | Registration |
|-------|-----------|--------------|
| `qwen3_vl` | `qwen3_vl/checkpoint_tensor_converter.py:Qwen3VLAttentionCheckpointTensorConverter` | `qwen3_vl/__init__.py` binds it as `staticmethod` on `Qwen3VLForConditionalGeneration / Qwen3VLModel` |
| `qwen3_vl_moe` | Composite in `qwen3_vl_moe/qkv_checkpoint_tensor_converter.py` wrapping (a) the existing MoE expert layout converter and (b) the QKV converter from `qwen3_vl` | `qwen3_vl_moe/__init__.py` binds the composite on `Qwen3VLMoeForConditionalGeneration / Qwen3VLMoeModel / Qwen3VLMoeTextModel` |

No user action is required. A partial / corrupted checkpoint (one of
`q/k/v_proj` missing for some attention layer) raises a loud
`RuntimeError` from `converter.finalize()` instead of silently producing
a model with uninitialized weights.

### 10.2 Saving a checkpoint (VeOmni → disk)

VeOmni training checkpoints emit the **fused** `qkv_proj.weight` key
directly to safetensors. This mirrors how Qwen3-MoE saves the fused
`gate_up_proj`/`down_proj` keys: the v5 fused layout becomes the
on-disk format. Two implications:

1. A VeOmni-saved Qwen3-VL checkpoint can be reloaded into another
   VeOmni run with the `ws_push` (or eager fallback) path with no
   conversion. The same converter from §10.1 is a no-op on already-fused
   keys because its regex only matches `q/k/v_proj.*` keys.
2. The on-disk format is **not** bit-compatible with vanilla HF
   `transformers` Qwen3-VL safetensors. Pushing to HF Hub / vLLM /
   SGLang requires running `scripts/qkv_split.py` once, mirroring the
   MoE-side `scripts/moe_ckpt_merge/moe_split.py` workflow.

**Standard export sequence**:

```bash
# Step 1: DCP -> HF safetensors (still fused; same step as MoE)
python scripts/merge_dcp_to_hf.py --load-dir <DCP> --save-dir <merged>

# Step 2: fused qkv_proj -> per-Linear q/k/v_proj
python scripts/qkv_split.py --merge_hf_path <merged> --split_hf_path <output>
```

`qkv_split.py` consumes `<prefix>.self_attn.qkv_proj.{weight,bias}` keys
and emits `q_proj` / `k_proj` / `v_proj` triplets split along `dim=0`
by `(n_q*hd, n_kv*hd, n_kv*hd)`. It rebuilds `model.safetensors.index.json`
and copies the config / tokenizer assets. The split is the exact inverse
of the load-side `Qwen3VLAttentionCheckpointTensorConverter`'s
`torch.cat`, so the output is round-trip bit-exact with:

- **VeOmni** — the load converter cats per-Linear back into `qkv_proj`
- **HuggingFace** `from_pretrained()`
- **vLLM / SGLang**

For in-process slice access without writing files, `FusedQKVLinear`
still exposes `q_weight_view()` / `k_weight_view()` / `v_weight_view()`
(and matching `*_bias_view()` helpers).

### 10.3 Optimizer state resume

The model-side converter only translates **model** state_dict keys.
Optimizer state (Adam `exp_avg / exp_avg_sq`, momentum buffers, etc.) is
keyed by parameter name and is **not** rewritten — so an old DCP
checkpoint whose optimizer state was saved against the three-Linear
layout (`q_proj.weight`, `k_proj.weight`, `v_proj.weight`) cannot be
resumed onto a model that exposes `qkv_proj.weight` as the parameter.
This matches the Qwen3-MoE posture: the recommended migration paths are

- **Re-warm from the HF model checkpoint** (no optimizer state) — the
  loader converter merges weights, optimizer state is reinitialized at
  the first step, training proceeds.
- **One-shot offline migration** — write a script that loads the old
  optimizer state, cats `(exp_avg_q, exp_avg_k, exp_avg_v)` along
  `dim=0` and saves under `qkv_proj` keys. The fused/HF mapping is the
  exact inverse of §10.2's split.

A built-in optimizer-state converter is not currently planned; if your
workflow requires resuming a long-running pre-`ws_push` job, the offline
script is the supported path.

---

## 11. Limitations & fallbacks

| Situation | Behavior |
|---|---|
| `ulysses_size == 1` | Callback self-disables (manager not allocated); eager path active |
| `num_kv_heads % ulysses_size != 0` | Callback construction raises — fused kernel cannot replicate KV heads |
| Non-Hopper GPU or no NVSHMEM | `_preflight_ws_push` raises at config-validation time |
| `(bs, local_seq, dtype)` differs from buffer config at a forward call | `mgr.is_compatible(...)` returns False → wrapper falls through to eager for that call (no crash) |
| User explicitly passes `ws_push_state=` and `W_qkv_B=` | Transparent dispatch skipped; caller in full control |
| `python -O` | All fused-branch precondition checks are explicit `raise ValueError`, so they survive `-O`-stripped runs (no `assert`s on the critical path) |

---

## 12. Memory budget & configuration checklist

The symmetric-memory buffer (`peer_out_buf`) is allocated **once** at
`on_train_begin` and sized exactly for the static
`(bs, local_seq, nheads_*, head_dim, dtype)` extracted from
`TrainingArguments`. It cannot be resized at runtime — `_symm_mem.rendezvous`
is a blocking collective and `WsPushState` is `frozen=True`. This makes
two things matter:

1. The buffer's HBM footprint is a deterministic function of your config —
   you can compute it before launching.
2. Every batch must arrive with **exactly** the `(bs, local_seq)` shape
   the buffer was sized for, or the transparent dispatcher silently falls
   back to eager.

### 12.1 HBM budget per rank

The symm-mem blob is laid out as `[peer_out | pad-to-128B | sync]` and
the peer_out region dominates everything else:

```
peer_out_nbytes  =  bs × seq_global × pack_hidden_local × itemsize
                 =  micro_batch_size
                  × max_seq_len           ← already implies × world_size / world_size
                  × (nheads_q + nheads_k + nheads_v) × head_dim / world_size
                  × itemsize              ← 2 bytes for bf16
```

Equivalently, **per-rank HBM = `micro_batch_size × max_seq_len × N_total × itemsize / world_size`**
where `N_total = (nheads_q + nheads_k + nheads_v) × head_dim`. Worked
examples:

| Model | `(Hq, Hk, Hv, hd)` | `bs` | `max_seq_len` | `ws` | `peer_out_buf` |
|---|---|--:|--:|--:|--:|
| Llama-7B (MHA) | (32, 8, 8, 128) | 1 | 8 K | 8 | **12 MB** |
| Llama-7B | (32, 8, 8, 128) | 1 | 32 K | 8 | 48 MB |
| Llama-7B | (32, 8, 8, 128) | 1 | 128 K | 8 | 192 MB |
| Llama-7B | (32, 8, 8, 128) | 1 | 256 K | 4 | **768 MB** |
| gpt-oss-120b shape (sweep test) | (64, 8, 8, 64) | 1 | 32 K | 8 | 80 MB |
| gpt-oss-120b shape | (64, 8, 8, 64) | 1 | 256 K | 8 | **640 MB** |
| gpt-oss-120b shape | (64, 8, 8, 64) | 1 | 256 K | 4 | **1.28 GB** |

A few takeaways:

- HBM scales **linearly** with `max_seq_len` and `micro_batch_size`, and
  **inversely** with `world_size`. So at fixed total work
  (`max_seq_len` constant) wider Ulysses → smaller per-rank buffer.
- There is **one buffer per process**, not per attention layer.
  `WsPushStateManager` is process-singleton and reused across all
  transformer blocks.
- This sits on top of whatever your model weights / activations /
  FSDP2 shard already use — budget it explicitly.

### 12.2 Configuration checklist

To make the fused kernel actually fire (and not silently fall back to
eager every step), set the following in your YAML:

```yaml
data:
  max_seq_len: 32768          # ← the absolute upper bound on training seq
                              #   length. The buffer is sized for THIS value.

train:
  micro_batch_size: 1         # ← static — must not change during training
  dyn_bsz: false              # ← dynamic batching produces [1, packed_seq]
                              #   tensors that don't match the (bs, local_seq)
                              #   buffer shape; fused kernel will silently
                              #   fall back to eager. Use static batching.

parallel:
  ulysses_size: 8

model:
  ops_implementation:
    ulysses_qkv_projection_implementation: ws_push  # default: "eager"
```

Hard requirements enforced by `FusedUlyssesStateCallback._prepare_init_kwargs`:

- `args.data.max_seq_len` must be set (otherwise `ValueError` at trainer
  construction time).
- `max_seq_len % ulysses_size == 0`.
- `num_kv_heads % ulysses_size == 0` (kernel cannot replicate KV heads).
- `ulysses_size > 1` (a 1-rank Ulysses group defeats the kernel).

Soft constraints surfaced as warnings (the callback emits
`logger.warning_rank0`):

- `args.train.dyn_bsz=True` → dynamic batching packs samples into
  `[1, packed_seq]` shapes that don't match the buffer. Most or all
  forwards will silently fall back to eager.
- `args.train.pad_to_length` set but `!= max_seq_len` → similar
  silent-fallback risk.

### 12.3 Sizing too small / too large

| Choice | Outcome |
|---|---|
| `max_seq_len` smaller than actual data | Every batch's seq exceeds `local_seq`; `is_compatible` returns False → silent eager fallback. Kernel never fires. |
| `max_seq_len` equal to data upper bound | Buffer minimal viable size; kernel hits on every step (given static batching). |
| `max_seq_len` larger than data upper bound | Buffer over-allocated → wastes HBM, but kernel still fires if your collator pads every batch to `max_seq_len`. |
| Variable seq across phases (e.g. curriculum) | Currently not supported by a single state. Either pick the maximum (kernel hits everywhere, wastes HBM in low-seq phases) or accept eager fallback for non-matching phases. A multi-shape manager is listed in [§15](#15-future-work). |

### 12.4 How to confirm the fused kernel is actually firing

The callback logs once at `on_train_begin` with the manager's full
shape tuple:

```
FusedUlyssesStateCallback: WsPushStateManager active (shape=(1, 4096, 64, 8, 8, 64, torch.bfloat16))
```

If your batches don't match this shape, every step takes the eager
path — same training result, but the symm-mem buffer is dead HBM. The
quickest sanity check is to wrap one forward with the profiler and look
for `ws_push_gemm_a2a` in the kernel trace; its absence means the
dispatcher is falling through.

---

## 13. Verification & tests

Two distributed test files cover the integration end-to-end (both
require ≥ 2 Hopper GPUs, NVSHMEM, and bf16):

- **`tests/parallel/ulysses/test_fused_ulysses_pre_attn.py`** — forward
  numeric parity between eager and `ws_push`, bf16 tolerance `5e-2`.
  Pre-existing; not modified.
- **`tests/parallel/ulysses/test_fused_ulysses_backward.py`** — forward
  **and backward** parity, ensures the combination of fused forward +
  unchanged hand-written backward matches eager grads on
  `q/k/v.weight`, `q/k/v.bias`, `norm_q/norm_k.weight`, and the input
  activation. New in this PR.

Two single-rank unit tests (no distributed init, importable on any
host):

- **`tests/ops/test_fused_ulysses_state_manager.py`** — OpSpec
  registration, `apply_global_ops` plumbing for both `eager` and
  `ws_push`, preflight on the local host, `WsPushStateManager` cache
  hit/miss/invalidate semantics, teardown idempotence.
- **`tests/ops/test_fused_ulysses_callback.py`** — callback gating
  paths under varying `ulysses_qkv_projection_implementation`,
  `ulysses_size`, and `model_config` (missing
  `num_attention_heads`, MHA fallback for `num_key_value_heads`, etc.).

Quick smoke run (single-rank, no distributed):

```bash
pixi run -- python3 -m pytest \
    tests/ops/test_fused_ulysses_state_manager.py \
    tests/ops/test_fused_ulysses_callback.py \
    -v
```

Multi-rank parity (2× Hopper required):

```bash
torchrun --nproc-per-node=2 \
    tests/parallel/ulysses/test_fused_ulysses_pre_attn.py
torchrun --nproc-per-node=2 \
    tests/parallel/ulysses/test_fused_ulysses_backward.py
```

---

## 14. Key files

| Path | Role |
|---|---|
| `veomni/ops/kernels/fused_ulysses_projection/_config.py` | `register_op(OpSpec(name="ulysses_qkv_projection", scope=OpScope.GLOBAL, ...))`; `_preflight_ws_push`; eager / ws_push side-effects |
| `veomni/ops/kernels/fused_ulysses_projection/state_manager.py` | `WsPushStateManager`; module-level `_active_impl_name`, `_active_manager`; `set/get/clear_active_manager`; per-layer cache + invalidate |
| `veomni/ops/kernels/fused_ulysses_projection/__init__.py` | Side-effect import of `_config`; re-export of manager helpers |
| `veomni/ops/kernels/fused_ulysses_projection/ulysses_pre_attn.py` | Host-side orchestration; `init_ws_push_state`; `ws_push_forward_impl` |
| `veomni/ops/kernels/fused_ulysses_projection/gemm_ws_packqkv.py` | CuTe DSL GEMM + PackQKV epilogue |
| `veomni/ops/kernels/__init__.py` | +1 line to side-effect-import `fused_ulysses_projection` |
| `veomni/arguments/arguments_types.py` | New field `ulysses_qkv_projection_implementation: str = "eager"`; `_NPU_ALLOWED` entry |
| `veomni/distributed/sequence_parallel/async_ulysses.py` | `assert` → `raise ValueError` on the fused branch; wrapper transparent dispatch |
| `veomni/trainer/callbacks/fused_ulysses_state_callback.py` | `FusedUlyssesStateCallback` (on_train_begin / on_step_end / on_train_end) |
| `veomni/trainer/callbacks/__init__.py` | Re-export `FusedUlyssesStateCallback` |
| `veomni/trainer/base.py` | Construct the callback in `_init_callbacks`; wire into the three dispatchers |
| `tests/parallel/ulysses/test_fused_ulysses_backward.py` | Forward + backward numeric parity (new) |
| `tests/ops/test_fused_ulysses_state_manager.py` | Single-rank unit tests (new) |
| `tests/ops/test_fused_ulysses_callback.py` | Single-rank unit tests (new) |

For background, see also:

- [`docs/key_features/ulysses.md`](./ulysses.md) — Ulysses SP fundamentals and Async Ulysses
- [`docs/design/kernel_selection.md`](../design/kernel_selection.md) — VeOmni's overall configurable-kernel framework

---

## 15. Future work

The following are explicitly out of scope for the present integration
and tracked as follow-ups:

- **Fused backward** mirroring the forward (today the backward still
  performs 3× `all_to_all_tensor` + 3× weight-grad GEMM).
- **`torch.compile` / aliasing-safe `custom_op` declaration**. The
  current `mutates_args=("peer_out_buf",)` on `ws_push_gemm_a2a` does
  not declare the peer-side writes performed by the epilogue, which
  matters for AOT dispatch / alias analysis.
- **Multi-shape / dynamic-batch manager** — keyed `dict[shape_key,
  WsPushState]` so packed-varlen or per-step shape changes don't fall
  back to eager.
- **Multi sp-group / multi-mesh** — today one manager per process.
- **Propagation of `assert → ValueError`** to other entry points inside
  `veomni/ops/kernels/fused_ulysses_projection/` (only the fused branch
  of `async_ulysses.py` was migrated in this PR).
