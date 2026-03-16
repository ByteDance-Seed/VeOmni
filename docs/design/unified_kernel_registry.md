# Unified Kernel Registry

## Status

**Proposal** | 2026-03-16

## Problem

VeOmni's current kernel selection is fragmented:

1. **Inconsistent config surface.** Attention and MoE use `OpsImplementationConfig`
   fields; RMSNorm/RoPE/SwiGLU use `VEOMNI_USE_LIGER_KERNEL` env var; loss uses
   `VEOMNI_USE_LIGER_KERNEL` + `VEOMNI_ENABLE_CHUNK_LOSS`. Users cannot
   independently select implementations per op.

2. **No extension point for internal users.** Internal teams ship custom kernels
   but must fork VeOmni OSS code to wire them in.

3. **No variant-aware validation.** Qwen3 MoE's RMSNorm is standard
   (`weight * x`); Qwen3.5 MoE's is offset (`(1+weight) * x`). Nothing prevents
   a user from selecting `liger` RMSNorm on Qwen3.5 MoE, producing silent
   incorrect results.

4. **Gaps in coverage.** Fused cross-entropy loss and MoE load-balancing
   auxiliary loss have no config-driven selection.

5. **HF `kernels` hub is insufficient.** It doesn't cover fused loss, MoE GEMM,
   partial RoPE, offset RMSNorm, or gated RMSNorm. We should not depend on it.

---

## Goals

- Single, explicit config surface: every replaceable op gets a field under
  `model.ops_implementation.*`
- Pluggable registry: internal users register kernels without modifying OSS code
- Variant-aware validation with hardware requirement checks
- Cover all ops including loss and MoE load-balancing
- Minimize diff from upstream transformers modeling code via patchgen

## Non-Goals

- Inference-only optimizations (PagedAttention, speculative decoding)
- NPU auto-selection beyond what already exists

---

## Design

### 1. Op Taxonomy

Every replaceable op is identified by an `(op_name, variant)` pair. The variant
encodes the mathematical semantics — two implementations are substitutable only
if they implement the same variant.

| `op_name` | Variants | Description |
|-----------|----------|-------------|
| `rms_norm` | `standard`, `offset_1` | Standard: `w * x/rms`. Offset: `(1+w) * x/rms` |
| `rms_norm_gated` | `standard` | `rms_norm(x) * silu(gate)` |
| `apply_rotary_pos_emb` | `full`, `partial` | Full: rotate all dims. Partial: rotate first `rotary_dim`, passthrough rest |
| `swiglu_mlp` | `standard` | SwiGLU MLP (gate/up/down) |
| `attention` | `standard` | Multi-head / GQA attention (existing `ALL_ATTENTION_FUNCTIONS` — unchanged) |
| `moe_experts` | `merged_gate_up`, `split_gate_up` | Expert GEMM dispatch |
| `cross_entropy_loss` | `standard` | Cross-entropy with optional fused logit projection |
| `moe_load_balancing_loss` | `standard` | Switch Transformer auxiliary loss |

### 2. Kernel Registry

```python
# veomni/ops/kernel_registry.py

@dataclass(frozen=True)
class HardwareRequirement:
    device_type: str                              # "cuda" | "npu"
    min_compute_capability: int | None = None     # e.g. 70, 80, 90

    def is_satisfied(self) -> bool:
        """Check against current runtime hardware."""
        ...

@dataclass(frozen=True)
class KernelSpec:
    name: str              # e.g. "liger", "triton_group_gemm"
    op_name: str           # e.g. "rms_norm"
    variant: str           # e.g. "standard"
    factory: callable      # () -> callable  (lazy import)
    hardware: HardwareRequirement
    description: str = ""

class KernelRegistry:
    """Global registry of kernel implementations.

    Keyed by (op_name, variant) -> {impl_name: KernelSpec}.
    "eager" is always implicitly available — it means "use the original HF
    code inline". It does not need to be registered.
    """

    def __init__(self):
        self._specs: dict[tuple[str, str], dict[str, KernelSpec]] = {}

    def register(self, spec: KernelSpec):
        key = (spec.op_name, spec.variant)
        self._specs.setdefault(key, {})
        if spec.name in self._specs[key]:
            raise ValueError(f"Duplicate kernel: {spec.name} for {key}")
        self._specs[key][spec.name] = spec

    def resolve(self, op_name: str, variant: str, impl_name: str) -> callable | None:
        """Resolve an implementation. Returns None for "eager".

        Raises KeyError if impl is not registered for this (op, variant).
        Raises RuntimeError if hardware requirements are not met.
        """
        if impl_name == "eager":
            return None
        key = (op_name, variant)
        specs = self._specs.get(key, {})
        if impl_name not in specs:
            available = ["eager"] + list(specs.keys())
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
        return spec.factory()

    def list_available(self, op_name: str, variant: str) -> list[str]:
        key = (op_name, variant)
        return ["eager"] + [
            name for name, spec in self._specs.get(key, {}).items()
            if spec.hardware.is_satisfied()
        ]

KERNEL_REGISTRY = KernelRegistry()
```

**OSS registrations** (in `veomni/ops/kernel_defaults.py`, imported at `veomni` init):

```python
from .kernel_registry import KERNEL_REGISTRY, KernelSpec, HardwareRequirement

# -- rms_norm (standard) --
KERNEL_REGISTRY.register(KernelSpec(
    name="liger",
    op_name="rms_norm", variant="standard",
    factory=lambda: __import__(
        "liger_kernel.transformers.rms_norm", fromlist=["LigerRMSNorm"]
    ).LigerRMSNorm,
    hardware=HardwareRequirement("cuda"),
))
# Note: no liger for rms_norm variant="offset_1" — only "eager" is available.

# -- apply_rotary_pos_emb (full) --
KERNEL_REGISTRY.register(KernelSpec(
    name="liger",
    op_name="apply_rotary_pos_emb", variant="full",
    factory=lambda: __import__(
        "liger_kernel.transformers.rope", fromlist=["liger_rotary_pos_emb"]
    ).liger_rotary_pos_emb,
    hardware=HardwareRequirement("cuda"),
))
# Note: no liger for variant="partial" — only "eager" is available.

# -- swiglu_mlp --
KERNEL_REGISTRY.register(KernelSpec(
    name="liger",
    op_name="swiglu_mlp", variant="standard",
    factory=lambda: __import__(
        "liger_kernel.transformers.swiglu", fromlist=["LigerSwiGLUMLP"]
    ).LigerSwiGLUMLP,
    hardware=HardwareRequirement("cuda"),
))

# -- moe_experts --
KERNEL_REGISTRY.register(KernelSpec(
    name="triton_group_gemm",
    op_name="moe_experts", variant="merged_gate_up",
    factory=lambda: __import__(
        "veomni.ops.fused_moe.group_gemm", fromlist=["group_gemm_fused_moe_forward"]
    ).group_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=70),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="quack_cutlass",
    op_name="moe_experts", variant="merged_gate_up",
    factory=lambda: __import__(
        "veomni.ops.fused_moe.quack_gemm", fromlist=["quack_gemm_fused_moe_forward"]
    ).quack_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=90),
))

# -- cross_entropy_loss --
KERNEL_REGISTRY.register(KernelSpec(
    name="liger_fused",
    op_name="cross_entropy_loss", variant="standard",
    factory=lambda: __import__(
        "veomni.ops.fused_cross_entropy.liger_kernel",
        fromlist=["fused_liger_kernel_cross_entropy"],
    ).fused_liger_kernel_cross_entropy,
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

Users select via YAML:

```yaml
model:
  ops_implementation:
    rms_norm_implementation: internal_fast_rmsnorm
```

### 3. Updated Config Surface

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

    # Per-op implementation selection (all default to "eager" = original HF code)
    rms_norm_implementation: str = "eager"
    rms_norm_gated_implementation: str = "eager"
    apply_rotary_pos_emb_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    moe_experts_implementation: str = "eager"
    cross_entropy_loss_implementation: str = "eager"
    moe_load_balancing_loss_implementation: str = "eager"
```

Convenience preset:

```yaml
model:
  ops_implementation:
    preset: liger   # expands to rms_norm=liger, rope=liger, swiglu=liger, loss=liger_fused
    moe_experts_implementation: triton_group_gemm  # override individual op
```

Preset expansion is best-effort: if the model's variant for an op has no `liger`
registration, that op stays `eager` (no error).

### 4. Patchgen Integration — Two Options

Both options use `veomni/patchgen/codegen.py`. The generated file is a
self-contained copy of the HF modeling file. The key question is how the
generated code dispatches between eager and fused kernels.

#### Option A: Inline Conditional Dispatch (Recommended)

The generated code **keeps the original HF eager implementation inline** and
adds a single `if` guard that calls the registry-resolved kernel when the user
selects a non-eager implementation. The variant declaration lives in this same
generated code — no separate `ModelOpsProfile` is needed.

**How it works:**

1. The patchgen config wraps each replaceable function/class with a thin
   conditional: if the resolved kernel is not None, call it; otherwise run the
   original HF code.

2. At `build_foundation_model` time, `KERNEL_REGISTRY.resolve()` is called for
   each op. The resolved callable (or None for eager) is stored on the model
   config as `config._veomni_ops["op_name"]`.

3. The variant is declared in the patchgen config as a string constant in the
   generated code. `build_foundation_model` reads it and passes it to
   `KERNEL_REGISTRY.resolve()`.

**Concrete example — `apply_rotary_pos_emb` in Qwen3.5 MoE:**

Patchgen config:

```python
# qwen3_5_moe_gpu_patch_gen_config.py

@config.replace_function(
    "apply_rotary_pos_emb",
    description="Add VeOmni kernel dispatch with eager fallback",
)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # Dispatch to registered kernel if available, else run original HF code.
    _kernel = _VEOMNI_OPS.get("apply_rotary_pos_emb")
    if _kernel is not None:
        return _kernel(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # --- original HF eager code (variant: partial) ---
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

Generated output (diff from upstream is just the 3-line guard at the top):

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py

# ======================================================================
# [PATCHED FUNCTION] apply_rotary_pos_emb
# Reason: Add VeOmni kernel dispatch with eager fallback
# ======================================================================
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    _kernel = _VEOMNI_OPS.get("apply_rotary_pos_emb")       # +
    if _kernel is not None:                                   # +
        return _kernel(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)  # +
    # --- original HF code below (unchanged) ---
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

**Concrete example — `Qwen3_5MoeExperts`:**

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py

# ======================================================================
# [PATCHED CLASS] Qwen3_5MoeExperts
# Reason: Add VeOmni kernel dispatch with eager fallback
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
        _kernel = _VEOMNI_OPS.get("moe_experts")                    # +
        if _kernel is not None:                                       # +
            return _kernel(                                           # +
                self, hidden_states, top_k_index, top_k_weights       # +
            )                                                         # +
        # --- original HF eager code below (unchanged) ---
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states
```

**Concrete example — `Qwen3_5MoeRMSNorm`:**

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py
# NOTE: NOT patched with dispatch guard. Qwen3.5 MoE uses offset_1 variant;
# no non-eager kernel is registered. The original HF code is emitted verbatim.

class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        # ... identical to upstream ...
```

**`_VEOMNI_OPS` binding** — a module-level dict in the generated file,
populated by `build_foundation_model`:

```python
# At top of generated file (added by patchgen config via add_post_import_block):
_VEOMNI_OPS: dict[str, callable] = {}

# Variant declarations — patchgen config emits these as constants:
_VEOMNI_OP_VARIANTS: dict[str, str] = {
    "rms_norm": "offset_1",
    "apply_rotary_pos_emb": "partial",
    "swiglu_mlp": "standard",
    "moe_experts": "merged_gate_up",
    "cross_entropy_loss": "standard",
    "moe_load_balancing_loss": "standard",
}
```

```python
# veomni/models/auto.py — build_foundation_model()

def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig):
    """Resolve kernels and populate the module's _VEOMNI_OPS dict."""
    variants = modeling_module._VEOMNI_OP_VARIANTS
    ops = modeling_module._VEOMNI_OPS
    for op_name, variant in variants.items():
        impl_name = getattr(ops_config, f"{op_name}_implementation", "eager")
        # resolve() returns None for "eager", raises on invalid/unsupported
        ops[op_name] = KERNEL_REGISTRY.resolve(op_name, variant, impl_name)
```

**Properties:**

- The generated code is self-documenting: you can read the function and see
  both the dispatch path and the full eager fallback.
- The variant declaration lives in the generated file (`_VEOMNI_OP_VARIANTS`),
  so each model's generated code is the single source of truth — no separate
  profile object to keep in sync.
- No global mutable state outside the generated module's own `_VEOMNI_OPS` dict.
- Internal users register new kernels via `KERNEL_REGISTRY.register()`;
  the generated code does not need to change.

#### Option B: Annotation + Runtime Forward Replacement

Instead of inline conditionals, mark replaceable points with decorators.
At `build_foundation_model` time, the resolved kernel is patched onto the
object via `setattr` / method replacement.

**Patchgen config** — annotate the function:

```python
@config.replace_function(
    "apply_rotary_pos_emb",
    description="Mark as replaceable via @veomni_replaceable",
)
@veomni_replaceable(op_name="apply_rotary_pos_emb", variant="partial")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # Original HF code — unchanged, no dispatch guard.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    ...
```

**Decorator definition:**

```python
# veomni/ops/replaceable.py

def veomni_replaceable(op_name: str, variant: str):
    """Mark a function/method as replaceable by the kernel registry.

    Stores metadata on the function. build_foundation_model() inspects
    these markers and swaps the function if a non-eager impl is selected.
    """
    def decorator(fn):
        fn._veomni_op_name = op_name
        fn._veomni_variant = variant
        return fn
    return decorator
```

**Runtime replacement** in `build_foundation_model`:

```python
def _apply_kernel_replacements(modeling_module, ops_config):
    """Walk the module namespace, find @veomni_replaceable markers, replace."""
    for name, obj in vars(modeling_module).items():
        if callable(obj) and hasattr(obj, "_veomni_op_name"):
            op_name = obj._veomni_op_name
            variant = obj._veomni_variant
            impl_name = getattr(ops_config, f"{op_name}_implementation", "eager")
            kernel = KERNEL_REGISTRY.resolve(op_name, variant, impl_name)
            if kernel is not None:
                setattr(modeling_module, name, kernel)

    # For class methods: walk classes, find decorated methods
    for name, cls in vars(modeling_module).items():
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            for attr_name, method in vars(cls).items():
                if callable(method) and hasattr(method, "_veomni_op_name"):
                    op_name = method._veomni_op_name
                    variant = method._veomni_variant
                    impl_name = getattr(ops_config, f"{op_name}_implementation", "eager")
                    kernel = KERNEL_REGISTRY.resolve(op_name, variant, impl_name)
                    if kernel is not None:
                        setattr(cls, attr_name, kernel)
```

For `nn.Module` classes like `Qwen3_5MoeRMSNorm` where the replacement is an
entire class (not just a method), use a class-level annotation:

```python
@veomni_replaceable_class(op_name="rms_norm", variant="offset_1")
class Qwen3_5MoeRMSNorm(nn.Module):
    # ... original HF code, unchanged ...
```

The replacement logic in `build_foundation_model` would swap the class in the
module namespace before model instantiation.

**Properties:**

- Zero diff in function/method bodies — the original HF code is completely
  untouched. Only a decorator is added.
- Replacement is a module-level `setattr`, which means it affects all instances
  of the model in the process (same trade-off as HF's `kernels` hub).
- Harder to debug: reading the generated code doesn't tell you what actually
  runs — you need to know what was patched at build time.
- Class replacement for `nn.Module` subclasses (RMSNorm, Experts) requires
  careful handling: the replacement class must have identical `__init__`
  signature and weight parameter names for checkpoint loading to work.

### 5. Option Comparison

| Aspect | Option A: Inline Conditional | Option B: Annotation + Runtime Replace |
|--------|------------------------------|----------------------------------------|
| **Diff from HF** | 3-line guard per op | 1-line decorator per op |
| **Readability** | Self-documenting — both paths visible inline | Must trace build-time patching to understand runtime behavior |
| **Internal extension** | Works out of the box — registry provides the callable | Works out of the box — same registry |
| **Global side effects** | None — dispatch dict is per-module | Module-level setattr affects all instances |
| **Class replacement** | Straightforward — eager code stays inline | Needs matching `__init__` signature and parameter names |
| **Debugging** | Set breakpoint inside the function, see the `if` | Must know replacement was applied |
| **`torch.compile` compatibility** | Guard is a simple dict lookup — graph-safe | Module-level function swap before compile — ok if done before `torch.compile()` |

**Recommendation: Option A.** The 3-line guard is a trivial diff, the code is
fully self-documenting, and there is no hidden mutation of module namespaces.

### 6. Loss and MoE LB Coverage

Both loss ops follow the same pattern as other ops.

#### Cross-Entropy Loss

Currently VeOmni patches `LOSS_MAPPING["ForCausalLM"]` and sets a global
`_cross_entropy` at import time. Under the new design, it is just another op
in the generated code.

In `ForConditionalGeneration.forward` (Qwen3.5 MoE example):

```python
# Patchgen keeps the existing override that calls self.loss_function(...),
# but build_foundation_model now binds it via the registry:
loss_impl = ops_config.cross_entropy_loss_implementation
loss_fn = KERNEL_REGISTRY.resolve("cross_entropy_loss", "standard", loss_impl)
if loss_fn is not None:
    model.loss_function = loss_fn
```

No change to the generated forward code — the existing
`self.loss_function(logits, labels, ...)` call already works.

#### MoE Load-Balancing Loss

`load_balancing_loss_func` is a standalone function. Same inline conditional
pattern as other ops:

```python
# In generated code:
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    _kernel = _VEOMNI_OPS.get("moe_load_balancing_loss")
    if _kernel is not None:
        return _kernel(gate_logits, num_experts, top_k, attention_mask)
    # --- original HF code below ---
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    ...
```

### 7. Full Example: Qwen3.5 MoE

Current state (from `qwen3_5_moe_gpu_patch_gen_config.py`):
- RMSNorm: not replaced (offset_1, incompatible with LigerRMSNorm) — **stays unchanged**
- RoPE: not replaced (partial, incompatible with liger) — **gets dispatch guard** (for future kernels)
- MoE: replaced with `PatchedQwen3_5MoeExperts` — **gets dispatch guard**
- Loss: overridden in `ForConditionalGeneration.forward` — **bound via registry**
- MoE LB loss: uses upstream unchanged — **gets dispatch guard**
- SwiGLU MLP: per-expert, not standalone — **not applicable for Qwen3.5 MoE**

**Patchgen config** (changes from current):

```python
# qwen3_5_moe_gpu_patch_gen_config.py

config = PatchConfig(
    source_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    target_file="patched_modeling_qwen3_5_moe_gpu.py",
)

# Emit variant declarations + dispatch dict at top of generated file
config.add_post_import_block("""
_VEOMNI_OPS: dict[str, callable] = {}
_VEOMNI_OP_VARIANTS: dict[str, str] = {
    "rms_norm": "offset_1",
    "apply_rotary_pos_emb": "partial",
    "moe_experts": "merged_gate_up",
    "cross_entropy_loss": "standard",
    "moe_load_balancing_loss": "standard",
}
""")

# ── RoPE: dispatch guard + original partial-rotary code ──────────────────
@config.replace_function(
    "apply_rotary_pos_emb",
    description="Add VeOmni kernel dispatch with eager fallback",
)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    _kernel = _VEOMNI_OPS.get("apply_rotary_pos_emb")
    if _kernel is not None:
        return _kernel(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ── MoE Experts: dispatch guard + original eager loop ────────────────────
@config.replace_class(
    "Qwen3_5MoeExperts",
    description="Add VeOmni kernel dispatch with eager fallback",
)
class PatchedQwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        _kernel = _VEOMNI_OPS.get("moe_experts")
        if _kernel is not None:
            return _kernel(self, hidden_states, top_k_index, top_k_weights)
        # --- original HF eager code ---
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


# ── MoE LB Loss: dispatch guard + original eager code ───────────────────
@config.replace_function(
    "load_balancing_loss_func",
    description="Add VeOmni kernel dispatch with eager fallback",
)
def load_balancing_loss_func(gate_logits, num_experts=None, top_k=2, attention_mask=None):
    _kernel = _VEOMNI_OPS.get("moe_load_balancing_loss")
    if _kernel is not None:
        return _kernel(gate_logits, num_experts, top_k, attention_mask)
    # --- original HF code (unchanged) ---
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    ...


# ── Remaining patches (SP, GatedDeltaNet, loss forward) — unchanged ──────
# ... same as current qwen3_5_moe_gpu_patch_gen_config.py ...
```

**User YAML:**

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    rms_norm_implementation: eager              # only eager available for offset_1
    apply_rotary_pos_emb_implementation: eager  # only eager available for partial
    moe_experts_implementation: triton_group_gemm
    cross_entropy_loss_implementation: liger_fused
    moe_load_balancing_loss_implementation: eager
```

**Validation at `build_foundation_model` time:**

```
# User mistakenly sets:
rms_norm_implementation: liger

# Error:
KeyError: No kernel 'liger' for op='rms_norm', variant='offset_1'.
Available: ['eager']
```

### 8. Lifecycle

```
import veomni                                     # (1) import time
  └─ import kernel_defaults                        #     register OSS kernels

(optional) import internal_kernels.register        # (2) internal registration
  └─ KERNEL_REGISTRY.register(...)                 #     add internal kernels

OpsImplementationConfig.__post_init__()            # (3) config parse time
  └─ rewrite attn_implementation for SP
  └─ expand preset if set

build_foundation_model(...)                        # (4) model build time
  ├─ import generated modeling module
  ├─ read _VEOMNI_OP_VARIANTS from module          #     variant declarations
  ├─ for each op: KERNEL_REGISTRY.resolve(          #     validate + resolve
  │       op, variant, user_impl)
  ├─ populate module._VEOMNI_OPS dict               #     bind resolved kernels
  ├─ bind model.loss_function if loss impl != eager
  └─ model init + weight loading

model.forward()                                    # (5) runtime
  ├─ apply_rotary_pos_emb: _VEOMNI_OPS lookup → eager or kernel
  ├─ attention: ALL_ATTENTION_FUNCTIONS[...]        #     unchanged
  ├─ moe_experts: _VEOMNI_OPS lookup → eager or kernel
  ├─ loss: self.loss_function(...)                  #     bound at step 4
  └─ moe_lb_loss: _VEOMNI_OPS lookup → eager or kernel
```

---

## Migration

| Current mechanism | New mechanism | Migration |
|---|---|---|
| `VEOMNI_USE_LIGER_KERNEL=1` env var | `rms_norm_implementation: liger` etc. | Deprecate env var; keep compat for 1 release |
| `gpu_patch.py` monkey-patching | patchgen inline conditionals | Remove `gpu_patch.py` files |
| `apply_veomni_loss_patch()` at import | `cross_entropy_loss_implementation` + build-time bind | Remove import-time patch |
| `apply_veomni_fused_moe_patch()` | `moe_experts_implementation` + registry | Remove standalone patch function |
| `moe_implementation` config field | `moe_experts_implementation` | Rename, keep alias for 1 release |

---

## Open Questions

1. **Preset system:** Should `preset: liger` silently skip ops where the
   model's variant has no `liger` registration, or warn?
2. **Attention:** Keep in `ALL_ATTENTION_FUNCTIONS` (shared with HF) or unify
   under `KERNEL_REGISTRY`?
3. **NPU auto-selection:** Should NPU be an explicit `npu_group_gemm`
   implementation name, or remain automatic?
4. **Multi-model processes:** `_VEOMNI_OPS` is per-module, so loading two
   different models in the same process works. But if two instances of the same
   model with different ops configs are needed, the module-level dict is shared.
   Is this a real use case?
