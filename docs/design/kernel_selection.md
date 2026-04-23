# Kernel Selection in VeOmni

VeOmni selects optimized kernel implementations for attention, cross-entropy
loss, Liger fused ops (RMSNorm, RoPE, SwiGLU), MoE, and load-balancing loss.
All selections are driven by config fields in `OpsImplementationConfig`.

## Quick Reference

Every configurable kernel lives under `model.ops_implementation.*` in YAML and
maps to a field on `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`).
Below is the full list — if a field is not in this table, it is not a kernel
selection knob.

| Kernel | Config field | Available values | Default | Selection time |
|--------|-------------|------------------|---------|----------------|
| Attention | `attn_implementation` | `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `native-sparse` | `"flash_attention_2"` | Config `__post_init__` + `build_foundation_model` |
| Cross-entropy loss | `cross_entropy_loss_implementation` | `eager`, `liger_kernel`, `npu` | `"eager"` | `apply_ops_config()` (before model build) |
| RMSNorm | `rms_norm_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3) | `"eager"` | Model registration via ops config singleton |
| SwiGLU MLP | `swiglu_mlp_implementation` | `eager`, `liger_kernel` | `"eager"` | Model registration via ops config singleton |
| Rotary embedding | `rotary_pos_emb_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3) | `"eager"` | Model registration via ops config singleton |
| Load-balancing loss | `load_balancing_loss_implementation` | `eager`, `triton` (CUDA `triton` or NPU `triton-ascend`) | `"eager"` | `apply_ops_config()` (before model build) |
| MoE experts | `moe_implementation` | `eager`, `fused_triton`, `fused_quack` (SM90+), `fused_npu` | `"eager"` | `build_foundation_model` |

The per-op fields are typed as plain `str` (not `Literal`), so third-party
backends can be registered via `extra_backends` in a model's `device_patch.py`
without modifying `OpsImplementationConfig`.

---

## Lifecycle Overview

```
import veomni                                 # (1) import time
  └─ apply_ops_patch()
       └─ apply_veomni_attention_patch()      # register FA2/3/4 with SP

OpsImplementationConfig.__post_init__()       # (2) config parse time
  ├─ validate requested backends are available
  ├─ rewrite attn_implementation for SP
  └─ set_ops_config(self)                     # populate singleton

BaseTrainer._build_model()                    # (3) model build time
  ├─ apply_ops_config(ops_implementation)     # install LOSS_MAPPING + GLOBAL patches
  │    ├─ install_loss_mapping(ce_impl)       # partial(ForCausalLMLoss, cross_entropy_fn=<impl>)
  │    └─ apply_global_ops(config)            # load_balancing_loss, etc.
  └─ build_foundation_model(...)
       ├─ apply_veomni_fused_moe_patch(...)   # bind MoE kernel
       ├─ device_patch.py reads ops config     # RMSNorm/RoPE/SwiGLU
       ├─ OpSlot.bind(impl_name)              # per-model OpSlot dispatch
       └─ model init + weight loading

model.forward()                               # (4) runtime
  ├─ attention: ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  ├─ loss: self.loss_function(...) -> LOSS_MAPPING[...] (pre-bound partial)
  │         OR veomni_causal_lm_loss(...) via OpSlot.use_non_eager_impl guard
  ├─ RMSNorm/RoPE/SwiGLU: Liger or HF default (set at registration)
  └─ MoE: fused_moe_forward(...) or eager loop
```

**Single install point.** `apply_ops_config` is the only place that binds
`LOSS_MAPPING` — there is no separate `apply_veomni_loss_patch` call. The
inner CE kernel (eager / liger / npu) is pre-bound onto the wrapper via
`functools.partial`, so runtime dispatch is just a function call and there
is no per-forward "which impl?" lookup.

**BaseTrainer contract.** Training is expected to go through `BaseTrainer`,
which calls `apply_ops_config` before `build_foundation_model`. Standalone
scripts (tests, eval harnesses) that call `build_foundation_model` directly
must call `apply_ops_config` themselves; if they forget, the model build
emits a warning and installs `OpsImplementationConfig()` defaults so
`self.loss_function` does not trip on VeOmni's extra `hidden_states=` /
`weights=` kwargs.

---

## 1. Attention

### Config

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2    # default
```

**Field:** `OpsImplementationConfig.attn_implementation`

### Available implementations

| Value | Kernel | Sequence Parallel | Requirements |
|-------|--------|:-:|---|
| `eager` | PyTorch | No | — |
| `sdpa` | `F.scaled_dot_product_attention` | No | — |
| `flash_attention_2` | Flash Attention v2 | Yes | `flash-attn` |
| `flash_attention_3` | Flash Attention v3 | Yes | `flash-attn-interface` |
| `flash_attention_4` | Flash Attention v4 | Yes | `flash-attn.cute` |
| `native-sparse` | Sparse attention | No | — |

When `MODELING_BACKEND=veomni` (the default), `__post_init__` automatically
rewrites `flash_attention_2/3/4` to VeOmni SP-aware variants
(`veomni_flash_attention_2_with_sp`, etc.) which wrap the underlying kernel
with DeepSpeed Ulysses sequence parallelism gather/scatter.

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Registration: `veomni/ops/kernels/attention/__init__.py` — `apply_veomni_attention_patch()`
- Plumbing: `veomni/models/auto.py` — `build_foundation_model(attn_implementation=...)`

---

## 2. Cross-Entropy Loss

### Config

```yaml
model:
  ops_implementation:
    cross_entropy_loss_implementation: eager   # or "liger_kernel", or custom str
```

**Field:** `OpsImplementationConfig.cross_entropy_loss_implementation`

### Available implementations

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | `fused_liger_kernel_cross_entropy` | `liger-kernel` package |
| `npu` | `chunk_loss_function` (chunked loss for `ForCausalLM` and `ForConditionalGeneration`; SP reduction handled internally) | `torch_npu` |
| `eager` | `eager_cross_entropy` (PyTorch `F.cross_entropy`) | — |

The `npu` chunk-loss binds only to `ForCausalLM` and
`ForConditionalGeneration`; `ForSequenceClassification` stays on
`eager_cross_entropy` because chunk_loss hard-codes the causal
`labels[..., 1:]` shift (incompatible with token-level classification
labels).

Selecting `liger_kernel` requires that the model's forward pass pass
`hidden_states=` and `weights=self.lm_head.weight` through
`self.loss_function(...)` — the Liger fused linear+CE kernel does the
projection itself and has no full logits tensor to fall back on. VeOmni's
patched modeling files (`patched_modeling_*.py`) already do this. If a model
whose forward was not patched calls `self.loss_function` without these
kwargs while `cross_entropy_loss_implementation="liger_kernel"`, the Liger
kernel raises `RuntimeError` with a pointer to the patch pattern — it does
**not** silently fall back to eager. Switch the field to `eager` if the
model cannot be patched.

### Key files

- Dispatch: `veomni/ops/kernels/cross_entropy/__init__.py` — `install_loss_mapping(impl)`
- Eager impl: `veomni/ops/kernels/cross_entropy/eager.py`
- Liger impl: `veomni/ops/kernels/cross_entropy/liger.py`
- NPU chunk loss: `veomni/ops/kernels/cross_entropy/chunk_loss.py` — `chunk_loss_function`

---

## 3. Per-Model Ops (RMSNorm, RoPE, SwiGLU MLP)

Each operation can be independently controlled. Despite the historical
"Liger fused ops" label, these fields are *not* Liger-only: they also accept
`npu` (for Ascend NPU backends) and `triton` (for model-specific Triton
kernels registered in the model's `device_patch.py`, e.g. DeepSeek-V3's
batch-invariant RMSNorm and deterministic RoPE).

### Config

```yaml
model:
  ops_implementation:
    rms_norm_implementation: eager             # eager | liger_kernel | npu | triton
    swiglu_mlp_implementation: eager           # eager | liger_kernel
    rotary_pos_emb_implementation: eager       # eager | liger_kernel | npu | triton
```

### Available implementations

#### `rms_norm_implementation`

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | `LigerRMSNorm` | `liger-kernel` package |
| `npu` | `torch_npu.npu_rms_norm` | `torch_npu` |
| `triton` | Model-specific Triton kernel registered via `extra_backends` (e.g. DeepSeek-V3 batch-invariant RMSNorm) | `triton`, per-model registration |
| `eager` | HuggingFace default (`{Model}RMSNorm`) | — |

#### `rotary_pos_emb_implementation`

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | `liger_rotary_pos_emb` | `liger-kernel` package |
| `npu` | `torch_npu.npu_rotary_mul` | `torch_npu` |
| `triton` | Model-specific Triton kernel registered via `extra_backends` (e.g. DeepSeek-V3 deterministic RoPE) | `triton`, per-model registration |
| `eager` | HuggingFace default (`apply_rotary_pos_emb`) | — |

#### `swiglu_mlp_implementation`

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | `LigerSwiGLUMLP` | `liger-kernel` package |
| `eager` | HuggingFace default (`{Model}MLP`) | — |

### What gets patched

For each selected backend, the model's `device_patch.py` either swaps the
target HF class (`replace_forward=False`) or rebinds its `forward`
(`replace_forward=True`). The summary of the Liger swap shape (the most
common case):

| Config field | Original | Liger replacement |
|---|---|---|
| `rms_norm_implementation` | `{Model}RMSNorm` | `LigerRMSNorm` |
| `rotary_pos_emb_implementation` | `apply_rotary_pos_emb` | `liger_rotary_pos_emb` |
| `swiglu_mlp_implementation` | `{Model}MLP` | `LigerSwiGLUMLP` |

The `npu` and `triton` backends follow the same `device_patch.py` flow — the
only difference is the kernel callable on the other side of the registry.

### Models with Liger support

Qwen2, Qwen3, Qwen3-MoE, Qwen2-VL, DeepSeek-V3, Llama, Seed-OSS.

### Key files

- Config singleton: `veomni/ops/config/singleton.py` — `get_ops_config()`, `set_ops_config()`
- Unified registry: `veomni/ops/config/registry.py` — `register_op()`, `apply_per_model_patches()`, `apply_global_ops()`
- OSS backend registration: `veomni/ops/kernels/{rms_norm,rotary,swiglu}/__init__.py`
- Per-model `extra_backends` (e.g. DeepSeek-V3 Triton): `veomni/models/transformers/{model}/device_patch.py`

---

## 4. Load-Balancing Loss

### Config

```yaml
model:
  ops_implementation:
    load_balancing_loss_implementation: eager   # or "triton", or custom str
```

**Field:** `OpsImplementationConfig.load_balancing_loss_implementation`

### Available implementations

| Value | Implementation | Requirements |
|-------|---------------|---|
| `triton` | Fused Triton kernel (`_load_balancing_loss` is rebound by `apply_ops_config` via the registry's `global_slot`) | `triton` on CUDA, or `triton-ascend` on Ascend NPU |
| `eager` | Pure-PyTorch reference (`load_balancing_loss_pytorch`) | — |

This is a `GLOBAL`-scope op: the function pointer
`veomni.ops.kernels.load_balancing_loss._load_balancing_loss` is rebound
once per process from `apply_ops_config()`, and every call site that
imports `from veomni.ops import load_balancing_loss_func` picks up the
selected backend automatically — no per-model patching needed.

### Key files

- Selection: `veomni/ops/kernels/load_balancing_loss/__init__.py` — `register_op(...)` entry
- Triton impl: `veomni/ops/kernels/load_balancing_loss/triton.py`
- Eager impl: `veomni/ops/kernels/load_balancing_loss/eager.py`

---

## 5. MoE Kernel

### Config

```yaml
model:
  ops_implementation:
    moe_implementation: fused_triton   # Triton group-gemm (GPU, SM70+)
    # moe_implementation: fused_quack  # Quack CUTLASS/CuTe (GPU, SM90+)
    # moe_implementation: fused_npu    # NPU group-gemm (Ascend)
    # moe_implementation: eager        # Reference PyTorch loop (very slow, debug only)
```

**Field:** `OpsImplementationConfig.moe_implementation`
**Default:** `"eager"`

The mode and kernel backend are expressed as a single field. Mismatches raise
at ``apply_veomni_fused_moe_patch`` time — no silent hardware fallback.

| Value | Kernel | Hardware | EP support |
|-------|--------|----------|:----------:|
| `eager` | PyTorch expert loop | Any | No |
| `fused_triton` | Triton group-gemm | GPU, SM70+ (V100+) | Yes |
| `fused_quack` | Quack CUTLASS/CuTe | GPU, SM90+ (H100+) | No |
| `fused_npu` | NPU group-gemm | Ascend NPU | Yes |

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Dispatch: `veomni/ops/kernels/moe/__init__.py` — `apply_veomni_fused_moe_patch()`
- Plumbing: `veomni/models/auto.py` — `build_foundation_model(moe_implementation=...)`

---

## Environment Variables

| Env var | Default | Scope | Notes |
|---------|---------|-------|-------|
| `MODELING_BACKEND` | `"veomni"` | Global | `"veomni"` or `"hf"` — controls whether VeOmni ops patches are applied |

Kernel selection is otherwise driven by `OpsImplementationConfig` fields.
The `VEOMNI_USE_LIGER_KERNEL` and `USE_GROUP_GEMM` environment variables
have been removed in favor of the per-op config fields.

All remaining env vars are registered in `veomni/utils/env.py` with defaults and can be
overridden by setting the corresponding shell environment variable.

---

## 5. Comparison with Transformers v5+ Kernel Selection

Transformers v5 (`transformers>=4.57`) introduces a unified kernel selection
framework that replaces the ad-hoc patching used in earlier versions.
This section compares VeOmni's approach (Sections 1-4 above) with the four
mechanisms available in Transformers v5, using `Qwen3MoE` and `Qwen3.5MoE` as
reference models.

### 5.1 Transformers v5 Mechanisms Overview

| # | Mechanism | Decorator / API | What it replaces | Scope |
|---|-----------|----------------|------------------|-------|
| 1 | Hub kernel layers | `@use_kernel_forward_from_hub("RMSNorm")` | `nn.Module.forward` | Per-class, via `kernels` library from HF Hub |
| 2 | Hub kernel functions | `@use_kernel_func_from_hub("rotary_pos_emb")` | Standalone functions (e.g. `apply_rotary_pos_emb`) | Per-function, via `kernels` library from HF Hub |
| 3 | Attention interface | `ALL_ATTENTION_FUNCTIONS.get_interface(...)` | Attention forward pass | Per-model via `config._attn_implementation` |
| 4 | Experts interface | `@use_experts_implementation` | MoE expert forward pass | Per-class via `config._experts_implementation` |

All four are defined in `transformers.integrations`:
- `hub_kernels.py` — mechanisms 1 & 2
- `moe.py` — mechanism 4
- `modeling_utils.py` — mechanism 3 (`ALL_ATTENTION_FUNCTIONS`)

### 5.2 Side-by-Side Comparison

#### RMSNorm

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | `gpu_patch.py` replaces `{Model}RMSNorm` class with `LigerRMSNorm` at import time | `@use_kernel_forward_from_hub("RMSNorm")` decorator on `Qwen3MoeRMSNorm`; at `model.kernelize()` time the `kernels` library downloads and swaps in `LigerRMSNorm` from `kernels-community/liger_kernels` |
| **Config** | `VEOMNI_USE_LIGER_KERNEL` env var | `USE_HUB_KERNELS` env var + `model.kernelize()` call |
| **When** | Model registration (import time) | Deferred — `kernelize()` after model init |
| **SP support** | N/A (norm is local) | N/A |
| **Qwen3.5 MoE gap** | Same as Qwen3 — Liger swap works | **Not annotated.** `Qwen3_5MoeRMSNorm` uses `weight * (1.0 + self.weight)` (offset-by-1 convention, weight init to zeros) instead of the standard `self.weight * x` (weight init to ones). No `@use_kernel_forward_from_hub("RMSNorm")` decorator. Standard `LigerRMSNorm` cannot replace it without accounting for the `+1.0` offset. |

#### Rotary Position Embedding (RoPE)

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | `gpu_patch.py` replaces `apply_rotary_pos_emb` function with `liger_rotary_pos_emb` at import time | `@use_kernel_func_from_hub("rotary_pos_emb")` on the `apply_rotary_pos_emb` function; `kernels` library downloads `apply_rotary_transformers` from `kernels-community/rotary`. The function is also attached to the Attention module via `@use_kernelized_func(apply_rotary_pos_emb)` so `kernelize()` can find it. |
| **Config** | `VEOMNI_USE_LIGER_KERNEL` env var | `USE_HUB_KERNELS` env var |
| **When** | Model registration (import time) | Import time (decorator) + `kernelize()` |
| **Qwen3.5 MoE gap** | N/A — VeOmni does not yet support Qwen3.5 MoE | **Partially annotated.** `apply_rotary_pos_emb` in `Qwen3_5MoeAttention` is annotated with `@use_kernelized_func` but **not** with `@use_kernel_func_from_hub("rotary_pos_emb")`. This is because Qwen3.5 MoE uses *partial RoPE* (`partial_rotary_factor < 1.0`): it splits Q/K into rotary and pass-through parts, applies RoPE only to the rotary part, then concatenates. The standard hub kernel `apply_rotary_transformers` does not handle this split-and-concat pattern. A dedicated partial-RoPE kernel could still be used. |

#### Attention

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | `apply_veomni_attention_patch()` registers SP-wrapped variants (`veomni_flash_attention_2_with_sp`, etc.) into `ALL_ATTENTION_FUNCTIONS` | Same `ALL_ATTENTION_FUNCTIONS` registry. Additionally supports hub-based attention kernels via `attn_implementation="kernels-community/flash-mla"` syntax (loaded by `load_and_register_attn_kernel()`). |
| **Config** | `OpsImplementationConfig.attn_implementation` | `config._attn_implementation` (set via `AutoModel.from_pretrained(attn_implementation=...)`) |
| **SP rewrite** | `__post_init__` rewrites `flash_attention_2` → `veomni_flash_attention_2_with_sp` | No SP support — upstream Transformers does not handle Ulysses SP |
| **Compatibility** | VeOmni registers into the **same** `ALL_ATTENTION_FUNCTIONS` registry that Transformers uses, so the two are compatible by design |

#### MoE Experts

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | `apply_veomni_fused_moe_patch()` replaces the global `fused_moe_forward` function pointer. Internally `config._moe_implementation ∈ {"eager", "fused"}` flags whether the patched experts should call the pointer at all; the actual kernel (Triton / Quack / NPU) is selected by `OpsImplementationConfig.moe_implementation`. | `@use_experts_implementation` decorator on `Qwen3MoeExperts` class; at forward time dispatches via `ALL_EXPERTS_FUNCTIONS.get_interface(config._experts_implementation, original_forward)`. Built-in implementations: `"batched_mm"` (BMM-based), `"grouped_mm"` (PyTorch `torch.nn.functional.grouped_mm`, requires PT 2.9+). |
| **Config** | `OpsImplementationConfig.moe_implementation` (`"eager"` / `"fused_triton"` / `"fused_quack"` / `"fused_npu"`) | `config._experts_implementation` (`"eager"` / `"batched_mm"` / `"grouped_mm"`) |
| **EP support** | `fused_triton` and `fused_npu` paths support Expert Parallelism via VeOmni's EP sharding | `batched_mm` handles invalid expert IDs (sentinel `>= num_experts`) for EP compatibility |
| **When** | Deferred to `build_foundation_model()` | Decorator at class definition time; dispatch at forward time |

**Note:** Transformers v5 hardcodes two MoE experts implementations (`batched_mm` and `grouped_mm`) and does not expose a registration interface for external fused kernels, so backends like VeOmni's Triton / Quack / NPU group-gemm must be plugged in through the `OpSlot` dispatch layer rather than via `ALL_EXPERTS_FUNCTIONS`.

### 5.3 Gaps — What Transformers v5 Does NOT Cover

The following areas have kernel selection in VeOmni but **no corresponding
mechanism** in Transformers v5:

#### 1. Fused Cross-Entropy Loss

Transformers v5 uses a `loss_function` property on `PreTrainedModel` that looks
up `LOSS_MAPPING[self.loss_type]` — this returns a standard PyTorch
`F.cross_entropy`-based loss. There is no decorator, no hub kernel, and no
env-var-based kernel swap for the loss function.

VeOmni replaces this at model-build time via `apply_ops_config(...)` →
`install_loss_mapping(impl)`, which binds `LOSS_MAPPING["ForCausalLM"]` to
`partial(ForCausalLMLoss, cross_entropy_fn=<impl>)` — where `<impl>` is
`fused_liger_kernel_cross_entropy` (GPU `liger_kernel`), `chunk_loss_function`
(NPU), or `eager_cross_entropy` (portable default). The fused Liger
cross-entropy computes the loss without materializing the full logits
tensor, which significantly reduces memory for large-vocabulary models.

**Implication:** When using VeOmni's trainer, the fused loss is transparent.
When using a standalone Transformers training loop, users would need to
call `apply_ops_config(OpsImplementationConfig(...))` themselves before
model construction (or directly monkey-patch `LOSS_MAPPING`).

#### 2. MoE Load-Balancing Auxiliary Loss

Both Qwen3MoE and Qwen3.5MoE in Transformers v5 include a standalone
`load_balancing_loss_func()` that computes the Switch Transformer auxiliary
loss. This function is called directly in `Qwen3MoeForCausalLM.forward()` —
there is no kernel selection, no registry, and no hub kernel for it.

VeOmni similarly does not provide a fused kernel for the auxiliary loss, but
this is worth noting because the load-balancing loss involves several
`one_hot → mean → dot` operations that could benefit from fusion, especially
at scale with many experts.

#### 3. Qwen3.5 MoE Variant-Specific Ops

Qwen3.5 MoE introduces architectural differences that prevent direct use of
the standard hub kernel annotations:

| Component | Qwen3 MoE | Qwen3.5 MoE | Why standard kernel fails |
|-----------|-----------|-------------|--------------------------|
| RMSNorm | `self.weight * x` (weight init ones) | `(1.0 + self.weight) * x` (weight init zeros) | LigerRMSNorm assumes no offset; applying it would produce incorrect results |
| RoPE | Full rotary on all dims | Partial rotary (`partial_rotary_factor`) — split, rotate, concat | Hub `apply_rotary_transformers` assumes full-dim rotation |
| RMSNormGated | N/A | `Qwen3_5MoeRMSNormGated` — norm then SiLU gate multiply | Uses explicit `fla` library selection (see below) |

**RMSNormGated: explicit `fla` library selection (not the hub kernel framework)**

Unlike RMSNorm and RoPE above, Qwen3.5 MoE's `RMSNormGated` **does** have a
fused kernel path — but it bypasses the Transformers v5 `@use_kernel_forward_from_hub`
framework entirely. Instead, `Qwen3_5MoeGatedDeltaNet.__init__` performs a
hard-coded conditional selection at model init time:

```python
# transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py

# At module top level:
if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

# In Qwen3_5MoeGatedDeltaNet.__init__:
self.norm = (
    Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
    if FusedRMSNormGated is None
    else FusedRMSNormGated(
        self.head_v_dim,
        eps=self.layer_norm_epsilon,
        activation=self.activation,
        device=torch.cuda.current_device(),
        dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
    )
)
```

This is a **5th kernel selection pattern** — not covered by any of the four
Transformers v5 mechanisms. It is a simple `if library_available else fallback`
check, similar to how the same file selects between `causal_conv1d_fn` (from
the `causal-conv1d` library) and a pure-PyTorch `torch_causal_conv1d_update`
fallback, and between `chunk_gated_delta_rule` (from `fla.ops`) and
`torch_chunk_gated_delta_rule`.

Key characteristics of this pattern:
- **No decorator, no registry, no env var** — purely hard-coded `if/else` in `__init__`
- **Library:** `flash-linear-attention` (`fla`) — a separate library from
  both Liger and the `kernels` hub
- **Scope:** Only the Gated DeltaNet linear attention layers in Qwen3.5 MoE;
  the standard full-attention `Qwen3_5MoeAttention` layers do not use this norm
- **Not configurable at runtime** — determined solely by whether `fla` is installed
- **`FusedRMSNormGated`** fuses the RMSNorm + SiLU gate multiply into a single
  Triton kernel, which the eager `Qwen3_5MoeRMSNormGated` does in two steps:
  `hidden = weight * (x / rms)` then `hidden = hidden * silu(gate)`

In Transformers v5, these remaining Qwen3.5 MoE ops (RMSNorm with `+1` offset,
partial RoPE) are left un-annotated — they always run the eager PyTorch
implementation. In theory, fused kernels could still be written for each (e.g.,
a Triton RMSNorm with `+1` offset, a partial-RoPE kernel), but no such kernels
currently exist in the `kernels-community` hub.



### 5.4 Summary Table

| Component | VeOmni mechanism | Transformers v5 mechanism | Compatible? | Gap |
|-----------|-----------------|--------------------------|:-----------:|-----|
| RMSNorm | `gpu_patch.py` Liger swap | `@use_kernel_forward_from_hub` | Parallel — both can apply | Qwen3.5 MoE `+1` offset norm not covered by either |
| RoPE | `gpu_patch.py` Liger swap | `@use_kernel_func_from_hub` + `@use_kernelized_func` | Parallel | Qwen3.5 MoE partial RoPE not covered by either |
| SwiGLU MLP | `gpu_patch.py` Liger swap | Not annotated in MoE models (MLP is per-expert, not standalone) | VeOmni only | — |
| Attention | `ALL_ATTENTION_FUNCTIONS` (shared registry) | `ALL_ATTENTION_FUNCTIONS` (same registry) | Yes | VeOmni adds SP wrapping |
| MoE experts | `apply_veomni_fused_moe_patch` (Triton/Quack) | `@use_experts_implementation` (batched_mm/grouped_mm) | No — different dispatch paths | VeOmni uses custom Triton kernels; HF uses PyTorch native `grouped_mm` |
| Cross-entropy | `apply_veomni_loss_patch` (Liger fused) | `LOSS_MAPPING` (standard `F.cross_entropy`) | VeOmni only | HF has no fused loss |
| MoE aux loss | Eager (same as HF) | Eager `load_balancing_loss_func` | Same | Neither provides a fused kernel |
| RMSNormGated | N/A | Hard-coded `fla.modules.FusedRMSNormGated` if `fla` installed, else eager (Qwen3.5 MoE only) | — | Bypasses all 4 HF v5 mechanisms; 5th ad-hoc pattern |

---

## Full Config Example

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused_triton
    cross_entropy_loss_implementation: liger_kernel
    rms_norm_implementation: liger_kernel
    swiglu_mlp_implementation: eager           # disable Liger for MLP only
    rotary_pos_emb_implementation: liger_kernel
    load_balancing_loss_implementation: triton
```
