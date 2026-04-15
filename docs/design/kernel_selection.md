# Kernel Selection in VeOmni

VeOmni selects optimized kernel implementations for attention, cross-entropy
loss, Liger fused ops (RMSNorm, RoPE, SwiGLU), MoE, and load-balancing loss.
All selections are driven by config fields in `OpsImplementationConfig`.

## Quick Reference

| Kernel | Config field | Default | Selection time |
|--------|-------------|---------|----------------|
| Attention | `attn_implementation` | `"flash_attention_2"` | Config `__post_init__` + `build_foundation_model` |
| Cross-entropy loss | `cross_entropy_loss_implementation` | `"eager"` | `apply_ops_config()` (before model build) |
| RMSNorm | `rms_norm_implementation` | `"eager"` | Model registration via ops config singleton |
| SwiGLU MLP | `swiglu_mlp_implementation` | `"eager"` | Model registration via ops config singleton |
| Rotary embedding | `rotary_pos_emb_implementation` | `"eager"` | Model registration via ops config singleton |
| Load-balancing loss | `load_balancing_loss_implementation` | `"eager"` | `apply_ops_config()` (before model build) |
| MoE implementation | `moe_implementation` | `None` | `build_foundation_model` |

All config fields live in `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`),
accessible via `model.ops_implementation.*` in YAML.

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
  ├─ apply_ops_config(ops_implementation)     # bind loss + LB patches
  └─ build_foundation_model(...)
       ├─ apply_veomni_fused_moe_patch(...)   # bind MoE kernel
       ├─ gpu_patch.py reads ops config       # RMSNorm/RoPE/SwiGLU
       └─ model init + weight loading

model.forward()                               # (4) runtime
  ├─ attention: ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  ├─ loss: _cross_entropy(...)
  ├─ RMSNorm/RoPE/SwiGLU: Liger or HF default (set at registration)
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
with DeepSpeed Ulysses sequence parallelism gather/scatter.

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Registration: `veomni/ops/flash_attn/__init__.py` — `apply_veomni_attention_patch()`
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
| `eager` | `eager_cross_entropy` (PyTorch `F.cross_entropy`) | — |

### Key files

- Selection: `veomni/ops/fused_cross_entropy/__init__.py` — `apply_veomni_loss_patch()`
- Eager impl: `veomni/ops/fused_cross_entropy/eager.py`
- Liger impl: `veomni/ops/fused_cross_entropy/liger_kernel.py`

---

## 3. Liger Fused Ops (RMSNorm, RoPE, SwiGLU MLP)

### Config

```yaml
model:
  ops_implementation:
    rms_norm_implementation: eager             # or "liger_kernel", or custom str
    swiglu_mlp_implementation: eager           # or "liger_kernel", or custom str
    rotary_pos_emb_implementation: eager       # or "liger_kernel", or custom str
```

Each operation can be independently controlled.

### What gets patched

When set to `"liger_kernel"`, each model's `gpu_patch.py` replaces
HuggingFace module classes on the corresponding operation:

| Config field | Original | Liger replacement |
|---|---|---|
| `rms_norm_implementation` | `{Model}RMSNorm` | `LigerRMSNorm` |
| `rotary_pos_emb_implementation` | `apply_rotary_pos_emb` | `liger_rotary_pos_emb` |
| `swiglu_mlp_implementation` | `{Model}MLP` | `LigerSwiGLUMLP` |

### Models with Liger support

Qwen2, Qwen3, Qwen3-MoE, Qwen2-VL, DeepSeek-V3, Llama, Seed-OSS.

### Key files

- Config singleton: `veomni/ops/ops_config.py` — `get_ops_config()`, `set_ops_config()`
- `veomni/models/transformers/{model}/gpu_patch.py` (7 model-specific files)

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
| `triton` | Fused Triton kernel | GPU |
| `eager` | PyTorch reference | — |

### Key files

- Selection: `veomni/ops/fused_load_balancing_loss/__init__.py`
- Triton impl: `veomni/ops/fused_load_balancing_loss/triton_kernel.py`
- Eager impl: `veomni/ops/fused_load_balancing_loss/torch_native.py`

---

## 5. MoE Kernel

### Config

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
| `fused` | Triton group-gemm | SM70+ (V100+) | Yes |
| `fused_quack` | Quack CUTLASS/CuTe | SM90+ (H100+) | No |
| *(NPU auto)* | NPU group-gemm | Ascend NPU | Yes |

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Dispatch: `veomni/ops/fused_moe/__init__.py` — `apply_veomni_fused_moe_patch()`
- Plumbing: `veomni/models/auto.py` — `build_foundation_model(moe_implementation=...)`

---

## Environment Variables

| Env var | Default | Scope | Notes |
|---------|---------|-------|-------|
| `MODELING_BACKEND` | `"veomni"` | Global | `"veomni"` or `"hf"` — controls whether VeOmni ops patches are applied |

Kernel selection is fully driven by `OpsImplementationConfig` fields.
The `VEOMNI_USE_LIGER_KERNEL` and `USE_GROUP_GEMM` environment variables
have been removed.

---

## Full Config Example

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused
    cross_entropy_loss_implementation: liger_kernel
    rms_norm_implementation: liger_kernel
    swiglu_mlp_implementation: eager           # disable Liger for MLP only
    rotary_pos_emb_implementation: liger_kernel
    load_balancing_loss_implementation: triton
```
