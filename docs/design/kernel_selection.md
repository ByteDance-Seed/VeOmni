# Kernel Selection in VeOmni

VeOmni selects optimized kernel implementations for attention, cross-entropy
loss, Liger fused ops, and MoE at different points in the lifecycle. This
document describes every selection mechanism, when it fires, and how to
configure it.

## Quick Reference

| Kernel | Config field | Env var | Default | Selection time |
|--------|-------------|---------|---------|----------------|
| Attention | `attn_implementation` | — | `"flash_attention_2"` | Config `__post_init__` + `build_foundation_model` |
| Cross-entropy loss | — | `VEOMNI_USE_LIGER_KERNEL` | `"1"` | Import time |
| Liger fused ops (RMSNorm, RoPE, SwiGLU) | — | `VEOMNI_USE_LIGER_KERNEL` | `"1"` | Model registration (import time) |
| MoE implementation | `moe_implementation` | — | `None` | `build_foundation_model` |

All config fields live in `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`),
accessible via `model.ops_implementation.*` in YAML.

---

## Lifecycle Overview

```
import veomni                                 # (1) import time
  └─ apply_ops_patch()
       ├─ apply_veomni_attention_patch()      # register FA2/3/4 with SP
       ├─ apply_veomni_loss_patch()           # bind cross-entropy kernel
       └─ (MoE patch is NOT applied here)

MODELING_REGISTRY.register()                  # (2) model class registration
  └─ gpu_patch files                          # Liger RMSNorm/RoPE/SwiGLU

OpsImplementationConfig.__post_init__()       # (3) config parse time
  └─ rewrite attn_implementation for SP

build_foundation_model(...)                   # (4) model build time
  ├─ apply_veomni_fused_moe_patch(backend=)   # bind MoE GEMM kernel
  ├─ config._moe_implementation = ...
  └─ model init + weight loading

model.forward()                               # (5) runtime
  ├─ attention: ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  ├─ loss: _cross_entropy(...)
  └─ MoE: fused_moe_forward(...) or eager loop
```

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
with DeepSpeed Ulysses sequence parallelism gather/scatter. This is why FA2/3/4
support SP — the rewrite is transparent to the user.

### Selection flow

1. **Config `__post_init__`** — `flash_attention_2` → `veomni_flash_attention_2_with_sp`
2. **`build_foundation_model`** — passed to HuggingFace `AutoModel.from_config(attn_implementation=...)`, stored as `config._attn_implementation`
3. **Import-time registration** — `apply_veomni_attention_patch()` registers the VeOmni names in `ALL_ATTENTION_FUNCTIONS`
4. **Forward** — Transformers dispatches to `flash_attention_forward()` via `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Registration: `veomni/ops/flash_attn/__init__.py` — `apply_veomni_attention_patch()`, `flash_attention_forward()`
- Plumbing: `veomni/models/auto.py` — `build_foundation_model(attn_implementation=...)`

---

## 2. Cross-Entropy Loss

### Config

No config field. Controlled by environment variable.

| Env var | Default | Values |
|---------|---------|--------|
| `VEOMNI_USE_LIGER_KERNEL` | `"1"` | `"0"` / `"1"` |
| `VEOMNI_ENABLE_CHUNK_LOSS` | `"0"` | `"0"` / `"1"` (NPU only) |

### Available implementations

| Implementation | When selected |
|---|---|
| `fused_liger_kernel_cross_entropy` | GPU + Liger installed + `VEOMNI_USE_LIGER_KERNEL=1` |
| `eager_cross_entropy` | GPU fallback, or NPU |
| `chunk_loss_function` | NPU + `VEOMNI_ENABLE_CHUNK_LOSS=1` |

### Selection flow

`apply_veomni_loss_patch()` runs at import time and sets the global
`_cross_entropy` function pointer:

1. NPU → `eager_cross_entropy` (+ optional `chunk_loss_function` for `LOSS_MAPPING`)
2. GPU + Liger + env `"1"` → `fused_liger_kernel_cross_entropy`
3. Fallback → `eager_cross_entropy`

### Key files

- Selection: `veomni/ops/fused_cross_entropy/__init__.py` — `apply_veomni_loss_patch()`
- Eager impl: `veomni/ops/fused_cross_entropy/eager.py`
- Liger impl: `veomni/ops/fused_cross_entropy/liger_kernel.py`

---

## 3. Liger Fused Ops (RMSNorm, RoPE, SwiGLU MLP)

### Config

No config field. Same environment variable as cross-entropy.

| Env var | Default |
|---------|---------|
| `VEOMNI_USE_LIGER_KERNEL` | `"1"` |

### What gets patched

When `VEOMNI_USE_LIGER_KERNEL=1` and the `liger_kernel` package is installed,
each model's `gpu_patch.py` replaces HuggingFace module classes:

| Component | Original | Liger replacement |
|---|---|---|
| RMSNorm | `{Model}RMSNorm` | `LigerRMSNorm` |
| Rotary embedding | `apply_rotary_pos_emb` | `liger_rotary_pos_emb` |
| SwiGLU MLP | `{Model}MLP` | `LigerSwiGLUMLP` |

### Selection flow

Patching happens at model class registration time (import of the model
module). Each model's `gpu_patch.py` checks:

```python
if is_liger_kernel_available() and get_env("VEOMNI_USE_LIGER_KERNEL") == "1":
    hf_module.apply_rotary_pos_emb = liger_rotary_pos_emb
    hf_module.ModelRMSNorm = LigerRMSNorm
    hf_module.ModelMLP = LigerSwiGLUMLP
```

### Models with Liger support

Qwen2, Qwen3, Qwen3-MoE, Qwen2-VL, DeepSeek-V3, Llama, Seed-OSS.

### Key files

- `veomni/models/transformers/{model}/gpu_patch.py` (7 model-specific files)

---

## 4. MoE Kernel

MoE kernel selection is controlled by a single `moe_implementation` field:

```yaml
model:
  ops_implementation:
    moe_implementation: fused          # Triton group-gemm (default fused path)
    # moe_implementation: fused_quack  # Quack CUTLASS/CuTe kernels (SM90+)
    # moe_implementation: eager        # Reference PyTorch loop (very slow, debug only)
```

**Field:** `OpsImplementationConfig.moe_implementation`
**Default:** `None` (falls back to `"eager"` per model config)

| Value | Kernel | Hardware | EP support |
|-------|--------|----------|:----------:|
| `eager` | PyTorch expert loop | Any | No |
| `fused` | Triton group-gemm (`group_gemm_same_nk`) | SM70+ (V100+) | Yes |
| `fused_quack` | Quack CUTLASS/CuTe (`quack.gemm_interface.gemm`) | SM90+ (H100+) | No |
| *(NPU auto)* | NPU group-gemm | Ascend NPU | Yes |

Models only see `_moe_implementation` as `"eager"` or `"fused"` — the
`fused_quack` variant is mapped to `"fused"` on the config, with the kernel
backend selected separately via `apply_veomni_fused_moe_patch`.

On NPU devices, the backend parameter is ignored — the NPU kernel is always
selected.

### Selection flow

Unlike attention and loss, the MoE patch is **not** applied at import time.
It is applied inside `build_foundation_model()`:

```python
def build_foundation_model(..., moe_implementation="fused_quack"):
    config._moe_implementation = "fused"
    apply_veomni_fused_moe_patch(moe_implementation="fused_quack")
```

This deferred approach allows the config to drive kernel selection without
env vars.

### Usage

**Via config (YAML):**

```yaml
model:
  ops_implementation:
    moe_implementation: fused_quack
```

**Via `build_foundation_model` (standalone scripts):**

```python
model = build_foundation_model(
    config_path="...",
    moe_implementation="fused_quack",
)
```

**Direct patch (tests / benchmarks):**

```python
from veomni.ops.fused_moe import apply_veomni_fused_moe_patch
apply_veomni_fused_moe_patch(moe_implementation="fused_quack")
```

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Dispatch: `veomni/ops/fused_moe/__init__.py` — `apply_veomni_fused_moe_patch()`
- Triton impl: `veomni/ops/fused_moe/group_gemm.py`
- Quack impl: `veomni/ops/fused_moe/quack_gemm.py`
- NPU impl: `veomni/ops/fused_moe/npu_group_gemm.py`
- Plumbing: `veomni/models/auto.py` — `build_foundation_model(moe_implementation=...)`

---

## Environment Variables Summary

| Env var | Default | Scope | Notes |
|---------|---------|-------|-------|
| `MODELING_BACKEND` | `"veomni"` | Global | `"veomni"` or `"hf"` — controls whether VeOmni ops patches are applied |
| `VEOMNI_USE_LIGER_KERNEL` | `"1"` | Global | Controls Liger kernel for RMSNorm/RoPE/SwiGLU + cross-entropy loss |
| `USE_GROUP_GEMM` | `"1"` | MoE | Gate for Triton group-gemm availability; set `"0"` to force fallback |
| `VEOMNI_ENABLE_CHUNK_LOSS` | `"0"` | NPU only | Enable chunked loss computation |

All env vars are registered in `veomni/utils/env.py` with defaults and can be
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
| **Mechanism** | `apply_veomni_fused_moe_patch()` replaces the global `fused_moe_forward` function pointer, keyed by `config._moe_implementation ∈ {"eager", "fused"}`. The actual GEMM backend (Triton vs Quack) is selected inside the patch function. | `@use_experts_implementation` decorator on `Qwen3MoeExperts` class; at forward time dispatches via `ALL_EXPERTS_FUNCTIONS.get_interface(config._experts_implementation, original_forward)`. Built-in implementations: `"batched_mm"` (BMM-based), `"grouped_mm"` (PyTorch `torch.nn.functional.grouped_mm`, requires PT 2.9+). |
| **Config** | `OpsImplementationConfig.moe_implementation` (`"eager"` / `"fused"` / `"fused_quack"`) | `config._experts_implementation` (`"eager"` / `"batched_mm"` / `"grouped_mm"`) |
| **EP support** | Triton `fused` path supports Expert Parallelism via VeOmni's EP sharding | `batched_mm` handles invalid expert IDs (sentinel `>= num_experts`) for EP compatibility |
| **When** | Deferred to `build_foundation_model()` | Decorator at class definition time; dispatch at forward time |

NOTE: transformers v5 moe hard codes 2 impl "batched_mm" and "grouped_mm" and does not provide a registration way for others to register other moe impl.

### 5.3 Gaps — What Transformers v5 Does NOT Cover

The following areas have kernel selection in VeOmni but **no corresponding
mechanism** in Transformers v5:

#### 1. Fused Cross-Entropy Loss

Transformers v5 uses a `loss_function` property on `PreTrainedModel` that looks
up `LOSS_MAPPING[self.loss_type]` — this returns a standard PyTorch
`F.cross_entropy`-based loss. There is no decorator, no hub kernel, and no
env-var-based kernel swap for the loss function.

VeOmni replaces this at import time with `apply_veomni_loss_patch()`, binding
either `fused_liger_kernel_cross_entropy` (GPU) or `chunk_loss_function` (NPU).
The fused Liger cross-entropy computes the loss without materializing the full
logits tensor, which significantly reduces memory for large-vocabulary models.

**Implication:** When using VeOmni's trainer, the fused loss is transparent.
When using a standalone Transformers training loop, users would need to manually
set `model.loss_function = custom_fused_loss` or monkey-patch `LOSS_MAPPING`.

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
