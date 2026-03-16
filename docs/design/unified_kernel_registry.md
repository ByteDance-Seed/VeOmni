# Unified Kernel Registry

## Status

**Proposal** | 2026-03-16

## Problem

VeOmni's current kernel selection is fragmented:

1. **Inconsistent config surface.** Attention and MoE use `OpsImplementationConfig` fields;
   RMSNorm/RoPE/SwiGLU use `VEOMNI_USE_LIGER_KERNEL` env var; loss uses
   `VEOMNI_USE_LIGER_KERNEL` + `VEOMNI_ENABLE_CHUNK_LOSS`. Users cannot
   independently select implementations per op.

2. **No extension point for internal users.** Internal teams ship custom
   kernels (e.g. faster MoE, custom fused norms) but must fork VeOmni OSS code
   to wire them in.

3. **No per-model op variant declaration.** Qwen3 MoE's RMSNorm is standard
   (`weight * x`); Qwen3.5 MoE's is offset (`(1+weight) * x`). There is no
   mechanism to declare which variant a model uses and which kernel
   implementations are compatible with that variant.

4. **Gaps in coverage.** Fused cross-entropy loss and MoE load-balancing
   auxiliary loss have no config-driven selection at all.

5. **HF `kernels` hub is insufficient.** It doesn't cover fused loss, MoE GEMM,
   partial RoPE, offset RMSNorm, or gated RMSNorm. We should not depend on it.

---

## Goals

- Single, explicit config surface: every replaceable op gets a field under
  `model.ops_implementation.*`
- Pluggable registry: internal users register kernels without modifying OSS code
- Per-model op variant declarations with hardware requirements
- Cover all ops including loss and MoE load-balancing
- Minimize diff from upstream transformers modeling code via patchgen

## Non-Goals

- Inference-only optimizations (e.g. PagedAttention, speculative decoding)
- NPU auto-selection beyond what already exists

---

## Design

### 1. Op Taxonomy

Every replaceable op is identified by a `(op_name, variant)` pair.

| `op_name` | Variants | Description |
|-----------|----------|-------------|
| `rms_norm` | `standard`, `offset_1` | Standard: `w * x/rms`. Offset: `(1+w) * x/rms` |
| `rms_norm_gated` | `standard` | `rms_norm(x) * silu(gate)` |
| `apply_rotary_pos_emb` | `full`, `partial` | Full: rotate all dims. Partial: rotate first `rotary_dim` dims, passthrough rest |
| `swiglu_mlp` | `standard` | SwiGLU MLP (gate/up/down) |
| `attention` | `standard` | Multi-head / GQA attention |
| `moe_experts` | `merged_gate_up`, `split_gate_up` | Expert GEMM dispatch |
| `cross_entropy_loss` | `standard` | Cross-entropy with optional fused logit projection |
| `moe_load_balancing_loss` | `standard` | Switch Transformer auxiliary loss |

### 2. Kernel Registry

```python
# veomni/ops/kernel_registry.py

from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareRequirement:
    """Declares what hardware a kernel implementation needs."""
    device_type: str                    # "cuda" | "npu"
    min_compute_capability: int | None = None  # e.g. 70 for V100, 80 for A100, 90 for H100

    def is_satisfied(self) -> bool:
        """Check against current runtime hardware."""
        ...

@dataclass(frozen=True)
class KernelSpec:
    """A registered kernel implementation."""
    name: str                           # e.g. "liger", "triton_group_gemm"
    op_name: str                        # e.g. "rms_norm"
    variant: str                        # e.g. "standard"
    factory: callable                   # () -> callable  (lazy import)
    hardware: HardwareRequirement
    description: str = ""

class KernelRegistry:
    """Global registry of kernel implementations."""

    def __init__(self):
        # (op_name, variant) -> {impl_name: KernelSpec}
        self._specs: dict[tuple[str, str], dict[str, KernelSpec]] = {}

    def register(self, spec: KernelSpec):
        key = (spec.op_name, spec.variant)
        self._specs.setdefault(key, {})
        if spec.name in self._specs[key]:
            raise ValueError(f"Duplicate kernel: {spec.name} for {key}")
        self._specs[key][spec.name] = spec

    def get(self, op_name: str, variant: str, impl_name: str) -> KernelSpec:
        key = (op_name, variant)
        specs = self._specs.get(key, {})
        if impl_name not in specs:
            available = list(specs.keys())
            raise KeyError(
                f"No kernel '{impl_name}' for op='{op_name}', variant='{variant}'. "
                f"Available: {available}"
            )
        spec = specs[impl_name]
        if not spec.hardware.is_satisfied():
            raise RuntimeError(
                f"Kernel '{impl_name}' requires {spec.hardware} "
                f"but current hardware does not satisfy it."
            )
        return spec

    def resolve(self, op_name: str, variant: str, impl_name: str) -> callable:
        """Get and instantiate the kernel. Called at build_foundation_model time."""
        spec = self.get(op_name, variant, impl_name)
        return spec.factory()

    def list_available(self, op_name: str, variant: str) -> list[str]:
        key = (op_name, variant)
        return [
            name for name, spec in self._specs.get(key, {}).items()
            if spec.hardware.is_satisfied()
        ]

KERNEL_REGISTRY = KernelRegistry()
```

**OSS registrations** (in `veomni/ops/kernel_defaults.py`, imported at `veomni` init):

```python
from .kernel_registry import KERNEL_REGISTRY, KernelSpec, HardwareRequirement

# -- rms_norm --
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="rms_norm", variant="standard",
    factory=lambda: None,  # use original HF code
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="liger",
    op_name="rms_norm", variant="standard",
    factory=lambda: __import__("liger_kernel.transformers.rms_norm", fromlist=["LigerRMSNorm"]).LigerRMSNorm,
    hardware=HardwareRequirement("cuda"),
))
# Qwen3.5-style offset RMSNorm — no liger replacement exists yet
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="rms_norm", variant="offset_1",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))

# -- apply_rotary_pos_emb --
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="apply_rotary_pos_emb", variant="full",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="liger",
    op_name="apply_rotary_pos_emb", variant="full",
    factory=lambda: __import__("liger_kernel.transformers.rope", fromlist=["liger_rotary_pos_emb"]).liger_rotary_pos_emb,
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="apply_rotary_pos_emb", variant="partial",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))

# -- moe_experts --
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="moe_experts", variant="merged_gate_up",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="triton_group_gemm",
    op_name="moe_experts", variant="merged_gate_up",
    factory=lambda: __import__("veomni.ops.fused_moe.group_gemm", fromlist=["group_gemm_fused_moe_forward"]).group_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=70),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="quack_cutlass",
    op_name="moe_experts", variant="merged_gate_up",
    factory=lambda: __import__("veomni.ops.fused_moe.quack_gemm", fromlist=["quack_gemm_fused_moe_forward"]).quack_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=90),
))

# -- cross_entropy_loss --
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="cross_entropy_loss", variant="standard",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="liger_fused",
    op_name="cross_entropy_loss", variant="standard",
    factory=lambda: __import__("veomni.ops.fused_cross_entropy.liger_kernel", fromlist=["fused_liger_kernel_cross_entropy"]).fused_liger_kernel_cross_entropy,
    hardware=HardwareRequirement("cuda"),
))

# -- moe_load_balancing_loss --
KERNEL_REGISTRY.register(KernelSpec(
    name="eager",
    op_name="moe_load_balancing_loss", variant="standard",
    factory=lambda: None,
    hardware=HardwareRequirement("cuda"),
))
```

**Internal registration** (in an internal package, never in OSS):

```python
# internal_kernels/register.py
from veomni.ops.kernel_registry import KERNEL_REGISTRY, KernelSpec, HardwareRequirement

KERNEL_REGISTRY.register(KernelSpec(
    name="internal_fast_rmsnorm",
    op_name="rms_norm", variant="standard",
    factory=lambda: ...,
    hardware=HardwareRequirement("cuda", min_compute_capability=80),
))
```

Users select it via YAML:

```yaml
model:
  ops_implementation:
    rms_norm_implementation: internal_fast_rmsnorm
```

### 3. Per-Model Op Variant Declaration

Each model declares which op variants it uses. This is a static declaration in
the model's patchgen config or `__init__.py`:

```python
# veomni/models/transformers/qwen3_5_moe/__init__.py

from veomni.ops.kernel_registry import ModelOpsProfile

QWEN3_5_MOE_OPS = ModelOpsProfile(
    model_name="qwen3_5_moe",
    ops={
        "rms_norm":                 "offset_1",   # (1+w)*x/rms
        "rms_norm_gated":           "standard",   # rms_norm(x) * silu(gate) — used in GatedDeltaNet
        "apply_rotary_pos_emb":     "partial",    # partial_rotary_factor=0.25
        "swiglu_mlp":               "standard",
        "attention":                "standard",
        "moe_experts":              "merged_gate_up",
        "cross_entropy_loss":       "standard",
        "moe_load_balancing_loss":  "standard",
    },
)

# For comparison:
QWEN3_MOE_OPS = ModelOpsProfile(
    model_name="qwen3_moe",
    ops={
        "rms_norm":                 "standard",   # w*x/rms
        "apply_rotary_pos_emb":     "full",       # full-dim rotation
        "swiglu_mlp":               "standard",
        "attention":                "standard",
        "moe_experts":              "merged_gate_up",
        "cross_entropy_loss":       "standard",
        "moe_load_balancing_loss":  "standard",
    },
)
```

At `build_foundation_model` time, VeOmni validates that every user-selected
implementation is registered for the model's declared variant:

```python
def validate_ops(profile: ModelOpsProfile, user_config: OpsImplementationConfig):
    for op_name, variant in profile.ops.items():
        impl_name = getattr(user_config, f"{op_name}_implementation", "eager")
        # This raises if impl is not registered or hardware is unsatisfied
        KERNEL_REGISTRY.get(op_name, variant, impl_name)
```

### 4. Updated Config Surface

```python
@dataclass
class OpsImplementationConfig:
    """model.ops_implementation.* — All kernel selections."""

    # Attention (existing — unchanged)
    attn_implementation: Literal[
        "eager", "sdpa",
        "flash_attention_2", "flash_attention_3", "flash_attention_4",
        "native-sparse",
    ] = "flash_attention_2"

    # NEW: per-op implementation selection
    rms_norm_implementation: str = "eager"
    apply_rotary_pos_emb_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    moe_experts_implementation: str = "eager"
    cross_entropy_loss_implementation: str = "eager"
    moe_load_balancing_loss_implementation: str = "eager"
    rms_norm_gated_implementation: str = "eager"
```

Convenience preset via YAML:

```yaml
model:
  ops_implementation:
    # Set all fused ops at once
    preset: liger  # expands to rms_norm=liger, rope=liger, swiglu=liger, loss=liger_fused
    # Override individual ops
    moe_experts_implementation: triton_group_gemm
```

### 5. Integration with Patchgen — Two Options

Both options use the existing `veomni/patchgen/codegen.py` infrastructure.
The generated file is a self-contained copy of the HF modeling file with
minimal diffs.

#### Option A: Runtime Forward Dispatch (Recommended)

At `build_foundation_model` time, resolve each op from the registry and bind it
to the model instance. The generated code calls a VeOmni dispatch wrapper.

**Patchgen config** — replace `apply_rotary_pos_emb` with a dispatch wrapper:

```python
# qwen3_5_moe_gpu_patch_gen_config.py

@config.replace_function(
    "apply_rotary_pos_emb",
    description="Dispatch to registry-selected RoPE kernel",
)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    return _veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
```

The wrapper `_veomni_apply_rotary_pos_emb` is a thin global function pointer
that gets bound at model build time:

```python
# veomni/ops/dispatch.py

# Global dispatch table — bound by build_foundation_model()
_DISPATCH: dict[str, callable] = {}

def bind_ops(profile: ModelOpsProfile, config: OpsImplementationConfig):
    """Resolve all ops from registry and bind to dispatch table."""
    for op_name, variant in profile.ops.items():
        impl_name = getattr(config, f"{op_name}_implementation", "eager")
        if impl_name == "eager":
            _DISPATCH[op_name] = None  # use original HF code
        else:
            _DISPATCH[op_name] = KERNEL_REGISTRY.resolve(op_name, variant, impl_name)

def _veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    fn = _DISPATCH.get("apply_rotary_pos_emb")
    if fn is not None:
        return fn(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # eager fallback — inline the original HF code
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    ...
```

**Diff from upstream:** Only the function/class body changes; signature and
call sites remain identical.

**Generated code example** for `Qwen3_5MoeExperts`:

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py

# ======================================================================
# [PATCHED CLASS] Qwen3_5MoeExperts
# Reason: Dispatch to registry-selected MoE kernel
# ======================================================================
class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Dispatch: if registry has a bound kernel, use it; else eager loop
        return veomni_moe_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )
```

Where `veomni_moe_experts_forward` checks `_DISPATCH["moe_experts"]`.

#### Option B: Patchgen-Time Conditional Codegen

Generate different code paths based on the known set of implementations. The
generated file contains `if/elif` branches.

```python
# In generated code:
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    _impl = _VEOMNI_OPS.get("apply_rotary_pos_emb")
    if _impl == "liger":
        return liger_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # eager (original HF code)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    ...
```

**Trade-off:** More code in the generated file but no global mutable state.
Less friendly to internal extension (new impls require re-running patchgen).

**Recommendation: Option A.** It keeps generated diffs minimal (one-line
wrapper call per op), and internal users can register implementations without
re-generating modeling code.

### 6. Loss and MoE LB Coverage

#### Cross-Entropy Loss

Currently VeOmni patches `LOSS_MAPPING["ForCausalLM"]` and sets a global
`_cross_entropy` at import time. Under the new design:

```python
# In build_foundation_model():
loss_impl = config.ops_implementation.cross_entropy_loss_implementation
loss_fn = KERNEL_REGISTRY.resolve("cross_entropy_loss", "standard", loss_impl)
if loss_fn is not None:
    model.loss_function = loss_fn
```

The patchgen config for `ForCausalLM.forward` / `ForConditionalGeneration.forward`
already overrides the forward to call `self.loss_function(...)`, so no
additional generated-code change is needed.

#### MoE Load-Balancing Loss

`load_balancing_loss_func` is a standalone function called in the model forward.
The patchgen config replaces it with a dispatch wrapper:

```python
@config.replace_function("load_balancing_loss_func")
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    return veomni_moe_lb_loss(gate_logits, num_experts, top_k, attention_mask)
```

Where `veomni_moe_lb_loss` checks `_DISPATCH["moe_load_balancing_loss"]` and
falls back to the original eager implementation.

### 7. Concrete Example: Qwen3.5 MoE

Current state (from `qwen3_5_moe_gpu_patch_gen_config.py`):
- RMSNorm: **not replaced** (offset_1 variant, incompatible with LigerRMSNorm)
- RoPE: **not replaced** (partial variant, incompatible with liger_rotary_pos_emb)
- MoE: replaced with `PatchedQwen3_5MoeExperts` using `_moe_implementation` flag
- Loss: overridden in `ForConditionalGeneration.forward` to call `self.loss_function`
- MoE LB loss: uses upstream `load_balancing_loss_func` unchanged

Under the new design:

```python
# veomni/models/transformers/qwen3_5_moe/__init__.py

QWEN3_5_MOE_OPS = ModelOpsProfile(
    model_name="qwen3_5_moe",
    ops={
        "rms_norm":                 "offset_1",
        "rms_norm_gated":           "standard",
        "apply_rotary_pos_emb":     "partial",
        "swiglu_mlp":               "standard",
        "attention":                "standard",
        "moe_experts":              "merged_gate_up",
        "cross_entropy_loss":       "standard",
        "moe_load_balancing_loss":  "standard",
    },
)
```

```yaml
# Config YAML
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    rms_norm_implementation: eager              # no liger for offset_1 yet
    apply_rotary_pos_emb_implementation: eager  # no liger for partial yet
    moe_experts_implementation: triton_group_gemm
    cross_entropy_loss_implementation: liger_fused
    moe_load_balancing_loss_implementation: eager
```

At `build_foundation_model("qwen3_5_moe", ...)`:
1. Load `QWEN3_5_MOE_OPS` profile
2. For each op, call `KERNEL_REGISTRY.get(op, variant, user_impl)` — validates
   compatibility and hardware
3. If user sets `rms_norm_implementation: liger` but model declares
   `variant="offset_1"`, and no `liger` is registered for `(rms_norm, offset_1)`,
   error is raised immediately with a clear message:
   ```
   KeyError: No kernel 'liger' for op='rms_norm', variant='offset_1'.
   Available: ['eager']
   Hint: Qwen3_5MoE uses offset-1 RMSNorm ((1+w)*x/rms) which is
   incompatible with standard LigerRMSNorm.
   ```
4. Bind resolved kernels to dispatch table
5. Proceed with model init and weight loading

**Patchgen config changes** — the existing `qwen3_5_moe_gpu_patch_gen_config.py`
changes are minimal:

```python
# BEFORE (current):
@config.replace_class("Qwen3_5MoeExperts", ...)
class PatchedQwen3_5MoeExperts(nn.Module):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self._moe_implementation == "fused":
            final_hidden_states = fused_moe_forward(...)
        elif self._moe_implementation == "eager":
            # ... 15 lines of eager loop ...

# AFTER (new):
@config.replace_class("Qwen3_5MoeExperts", ...)
class PatchedQwen3_5MoeExperts(nn.Module):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return veomni_moe_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )
```

The dispatch logic moves from generated code into `veomni/ops/dispatch.py`,
making the generated diff smaller and the dispatch logic reusable across models.

### 8. Lifecycle (Updated)

```
import veomni                                     # (1) import time
  └─ import kernel_defaults                        #     register OSS kernels
  └─ apply_ops_patch()
       └─ apply_veomni_attention_patch()           #     register FA2/3/4 with SP

(optional) import internal_kernels.register        # (2) internal registration
  └─ KERNEL_REGISTRY.register(...)                 #     add internal kernels

OpsImplementationConfig.__post_init__()            # (3) config parse time
  └─ rewrite attn_implementation for SP

build_foundation_model(...)                        # (4) model build time
  ├─ load ModelOpsProfile for this model
  ├─ validate_ops(profile, config)                 #     check all impls exist + hw ok
  ├─ bind_ops(profile, config)                     #     resolve and bind to dispatch table
  └─ model init + weight loading

model.forward()                                    # (5) runtime
  ├─ apply_rotary_pos_emb → _DISPATCH lookup
  ├─ attention: ALL_ATTENTION_FUNCTIONS[...]        #     unchanged
  ├─ moe_experts → _DISPATCH lookup
  ├─ loss → model.loss_function (bound at step 4)
  └─ moe_lb_loss → _DISPATCH lookup
```

---

## Migration

| Current mechanism | New mechanism | Migration |
|---|---|---|
| `VEOMNI_USE_LIGER_KERNEL=1` env var | `rms_norm_implementation: liger` etc. | Deprecate env var; keep backward compat for 1 release |
| `gpu_patch.py` monkey-patching | patchgen + dispatch wrappers | Remove `gpu_patch.py` files |
| `apply_veomni_loss_patch()` at import | `cross_entropy_loss_implementation` config field + bind at build time | Remove import-time patch |
| `apply_veomni_fused_moe_patch()` | `moe_experts_implementation` config field + registry | Remove standalone patch function |
| `moe_implementation` config field | `moe_experts_implementation` config field | Rename, keep alias for 1 release |

---

## Open Questions

1. **Preset system:** Should `preset: liger` auto-expand only for ops where the
   model's variant has a `liger` registration, or error if any op is missing?
2. **Attention in registry vs. `ALL_ATTENTION_FUNCTIONS`:** Attention already has
   a working registry shared with HF. Should we keep it separate or unify under
   `KERNEL_REGISTRY`?
3. **NPU auto-selection:** Currently NPU always overrides to its own kernel.
   Should this be an explicit `npu_group_gemm` implementation name, or remain
   automatic?
