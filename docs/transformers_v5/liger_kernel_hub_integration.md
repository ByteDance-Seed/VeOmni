# Liger Kernel Hub Integration

This document explains how VeOmni integrates [Liger Kernel](https://github.com/linkedin/Liger-Kernel) optimizations via the HuggingFace Transformers v5 **kernels-hub** system, and how to add support for a new model.

## Overview

In Transformers v5, the modeling code uses decorators like `@use_kernel_func_from_hub` and `@use_kernel_forward_from_hub` to mark functions and layers as replaceable by optimized kernel implementations. VeOmni leverages this mechanism to inject Liger Kernel fused operators (e.g. fused RMSNorm, fused RoPE) at model build time.

## How `use_liger` Works End-to-End

### 1. Configuration

Set `use_liger: true` in the YAML config or via CLI:

```yaml
# configs/text/qwen3.yaml
model:
  ops_implementation:
    use_liger: true
```

```bash
# or via CLI override
--model.ops_implementation.use_liger true
```

The parameter is defined in `veomni/arguments/arguments_types.py` as part of `OpsImplementationConfig`.

### 2. Trainer passes the flag

`BaseTrainer._build_model()` reads `self.args.model.ops_implementation.use_liger` and passes it to `build_foundation_model()`.

### 3. `build_foundation_model()` in `veomni/models/auto.py`

After the model is loaded, if `use_liger=True`:

1. Looks up the model's `config.model_type` (e.g. `"qwen3"`) in `LIGER_KERNEL_MAPPING_REGISTRY`.
2. Calls the registered factory to get a `KERNEL_MAPPING` dict.
3. Calls `_build_liger_kernel_mapping()` to resolve each entry into a `LocalLayerRepository` or `LocalFuncRepository`.
4. Applies the mapping with `kernels.kernelize(model, ...)`.

```python
# Simplified flow in auto.py
kernel_mapping = LIGER_KERNEL_MAPPING_REGISTRY[model_type]()
resolved_mapping = _build_liger_kernel_mapping(kernel_mapping)

with use_kernel_mapping(resolved_mapping, inherit_mapping=False):
    kernelize(model, mode=mode, device="cuda")
```

### 4. `_build_liger_kernel_mapping()` in `veomni/models/auto.py`

This function converts each entry in the `KERNEL_MAPPING` dict into a kernels-hub compatible repository object:

- `"type": "layer"` entries become `LocalLayerRepository` — used for `nn.Module` subclasses whose `forward()` replaces the original (matched by `@use_kernel_forward_from_hub`).
- `"type": "func"` entries become `LocalFuncRepository` — used for standalone functions (matched by `@use_kernel_func_from_hub`).

The module path is resolved via `importlib.import_module()`, and the repository is pointed at the correct directory and package name.

### A note on the hardcoded `"cuda"` device key

The resolved mapping uses `"cuda"` as the device key:

```python
resolved[entry_name] = {"cuda": repo}
```

This is **not** a hardware restriction. The kernels-hub library uses device keys to organize repositories, but `kernelize()` only checks the key — it does not enforce that the actual hardware matches. The `"cuda"` key works correctly on all supported backends including `"mps"`, `"npu"`, `"rocm"`, and `"xpu"`. We hardcode `"cuda"` as a simple way to bypass the device-type check during meta-init (where no real device is present). There is no need to change this for other hardware backends.

## Adding Liger Kernel Support for a New Model

Use Qwen3 as the reference implementation. You need to create three things:

### Step 1: Ensure the patched modeling file has the right decorators

The generated/patched modeling file (e.g. `generated/patched_modeling_<model>_gpu.py`) must use decorators from `transformers.integrations` to mark which classes and functions are replaceable by kernels. **Every entry you plan to add in `KERNEL_MAPPING` (Step 3) must have a corresponding decorator registered in the modeling file first — otherwise `kernelize()` will raise an error.**

```python
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)

@use_kernel_forward_from_hub("RMSNorm")
class MyRMSNorm(nn.Module):
    ...

@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    ...
```

These decorators are what the `kernels.kernelize()` call matches against when applying the mapping.

### Step 2: Create the `liger_kernels_hub/` wrapper package

Create a directory `veomni/models/transformers/<model>/liger_kernels_hub/` with:

- **`__init__.py`** — re-exports wrappers from `layers.py` so the kernel mapping can reference the package directly.
- **`layers.py`** — all stateless wrappers. The file **must** be named `layers.py` because `LocalLayerRepository` (used by `@use_kernel_forward_from_hub`) searches for layer classes in a file with this name.

```python
# layers.py
import torch.nn as nn
from liger_kernel.transformers.rms_norm import LigerRMSNorm as _OrigLigerRMSNorm

class LigerRMSNorm(nn.Module):
    """Stateless wrapper — kernels-hub requires no custom __init__."""
    def forward(self, hidden_states):
        return _OrigLigerRMSNorm.forward(self, hidden_states)
```

**Why wrappers?** Two reasons:

1. The `kernels` library expects replacement layers to be **stateless** `nn.Module` subclasses with no custom `__init__` or extra members. The wrapper delegates to the real Liger forward.
2. The model's patched code may have a **different function signature** than the upstream Liger kernel. For example, Qwen3's `apply_rotary_pos_emb` takes 5 arguments `(q, k, cos, sin, unsqueeze_dim=1)` while the Liger kernel takes 6 `(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)`. The wrapper bridges this gap:

```python
from liger_kernel.transformers.rope import liger_rotary_pos_emb as _orig

def liger_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    return _orig(q, k, cos, sin, position_ids=None, unsqueeze_dim=unsqueeze_dim)
```

### Step 3: Create the kernel mapping

Create `veomni/models/transformers/<model>/liger_kernel_mapping.py`. Each entry key (e.g. `"RMSNorm"`, `"rotary_pos_emb"`) **must** match the string used in the corresponding `@use_kernel_forward_from_hub` or `@use_kernel_func_from_hub` decorator added in Step 1. Adding an entry here without a matching decorator in the modeling file will cause `kernelize()` to raise an error.

```python
from ...loader import LIGER_KERNEL_MAPPING_REGISTRY

KERNEL_MAPPING = {
    "RMSNorm": {
        "type": "layer",
        "module": "veomni.models.transformers.<model>.liger_kernels_hub",
        "name": "LigerRMSNorm",
    },
    "rotary_pos_emb": {
        "type": "func",
        "module": "veomni.models.transformers.<model>.liger_kernels_hub",
        "name": "liger_rotary_pos_emb",
    },
}


@LIGER_KERNEL_MAPPING_REGISTRY.register("<model_type>")
def get_liger_kernel_mapping():
    return KERNEL_MAPPING
```

The `"<model_type>"` string must match `config.model_type` from the HuggingFace model config.

Each entry has three fields:

| Field | Description |
|-------|-------------|
| `type` | `"layer"` for `nn.Module` replacements, `"func"` for function replacements |
| `module` | Dotted Python import path to the module containing the kernel |
| `name` | Class or function name within that module |

### Step 4: Register the mapping via import

In `veomni/models/transformers/<model>/__init__.py`, add a side-effect import so the mapping is registered when the model package is loaded:

```python
from . import liger_kernel_mapping as _liger_kernel_mapping  # noqa: F401
```

## Checklist

- [ ] Patched modeling file uses `@use_kernel_forward_from_hub` / `@use_kernel_func_from_hub` with matching keys
- [ ] `liger_kernels_hub/__init__.py` — re-exports wrappers from `layers.py`
- [ ] `liger_kernels_hub/layers.py` — wrapper classes/functions with correct signatures
- [ ] `liger_kernel_mapping.py` — `KERNEL_MAPPING` entries match the decorators in the modeling file
- [ ] `__init__.py` — side-effect import of `liger_kernel_mapping`
- [ ] Config YAML sets `use_liger: true`

## Reference: Qwen3 File Layout

```
veomni/models/transformers/qwen3/
├── __init__.py                          # imports liger_kernel_mapping
├── liger_kernel_mapping.py              # registers "qwen3" in LIGER_KERNEL_MAPPING_REGISTRY
├── liger_kernels_hub/
│   ├── __init__.py                      # re-exports from layers.py
│   └── layers.py                        # stateless wrappers (LigerRMSNorm, liger_rotary_pos_emb)
└── generated/
    └── patched_modeling_qwen3_gpu.py     # uses @use_kernel_*_from_hub decorators
```
