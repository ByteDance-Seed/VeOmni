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
