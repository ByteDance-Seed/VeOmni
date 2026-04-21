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
| `rms_norm` | `standard`, `qwen3_5` | Standard: `w * x/rms`. Qwen3.5: `(1+w) * x/rms` (weight init zeros) |
| `rms_norm_gated` | `standard` | `rms_norm(x) * silu(gate)` |
| `apply_rotary_pos_emb` | `full`, `partial` | Full: rotate all dims. Partial: rotate first `rotary_dim`, passthrough rest |
| `swiglu_mlp` | `standard` | SwiGLU MLP (gate/up/down) |
| `attention` | `standard` | Multi-head / GQA attention (existing `ALL_ATTENTION_FUNCTIONS` — unchanged) |
| `moe_experts` | `standard` | Expert GEMM dispatch (merged gate+up projection, HF v5 convention) |
| `cross_entropy_loss` | `causal`, `seq_cls` | Causal-LM CE (shifts labels) vs. sequence-classification CE (no shift) |
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
# Note: no liger for rms_norm variant="qwen3_5" — only "eager" is available.

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
    op_name="moe_experts", variant="standard",
    factory=lambda: __import__(
        "veomni.ops.fused_moe.group_gemm", fromlist=["group_gemm_fused_moe_forward"]
    ).group_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=70),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="quack_cutlass",
    op_name="moe_experts", variant="standard",
    factory=lambda: __import__(
        "veomni.ops.fused_moe.quack_gemm", fromlist=["quack_gemm_fused_moe_forward"]
    ).quack_gemm_fused_moe_forward,
    hardware=HardwareRequirement("cuda", min_compute_capability=90),
))

# -- cross_entropy_loss (split by task to avoid mixing causal-LM label
#    shifting with sequence-classification token-level labels) --
KERNEL_REGISTRY.register(KernelSpec(
    name="liger_kernel",
    op_name="cross_entropy_loss", variant="causal",
    factory=_liger_fused_ce_causal_factory,   # partial(ForCausalLMLoss, cross_entropy_fn=liger)
    hardware=HardwareRequirement("cuda"),
))
KERNEL_REGISTRY.register(KernelSpec(
    name="liger_kernel",
    op_name="cross_entropy_loss", variant="seq_cls",
    factory=_liger_fused_ce_seq_cls_factory,  # partial(ForSequenceClassificationLoss, cross_entropy_fn=liger)
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
    rotary_pos_emb_implementation: str = "eager"
    swiglu_mlp_implementation: str = "eager"
    # MoE: mode ("eager" | "fused") + backend kernel ("triton" | "quack")
    moe_implementation: Literal["eager", "fused"] = "eager"
    fused_moe_kernel: Literal["triton", "quack"] = "triton"
    cross_entropy_loss_implementation: str = "eager"
    moe_load_balancing_loss_implementation: str = "eager"
```

Convenience preset:

```yaml
model:
  ops_implementation:
    preset: liger   # expands to rms_norm=liger, rope=liger, swiglu=liger, loss=liger_fused
    moe_implementation: fused       # override individual op
    fused_moe_kernel: triton
```

Preset expansion is best-effort: if the model's variant for an op has no `liger`
registration, that op stays `eager` (no error).

### 4. Op Slots

Each replaceable op gets a lightweight slot object. The slot holds only the
kernel binding (or None for eager). It does **not** hold the eager fallback —
the original HF code stays in place and the caller falls through to it via an
if-else. This keeps the diff from upstream HF minimal.

```python
# veomni/ops/dispatch.py

class OpSlot:
    """A slot for an optional kernel replacement.

    Created at module level in the generated file when the module is first
    imported. Starts unbound (_kernel = None). build_foundation_model()
    later calls .bind() to resolve a kernel from the registry.
    """
    def __init__(self, op_name: str, variant: str):
        self.op_name = op_name
        self.variant = variant
        self._kernel: callable | None = None

    def bind(self, impl_name: str):
        """Resolve from registry and bind. Called by build_foundation_model."""
        self._kernel = KERNEL_REGISTRY.resolve(self.op_name, self.variant, impl_name)

    @property
    def has_kernel(self) -> bool:
        return self._kernel is not None

    def __call__(self, *args, **kwargs):
        return self._kernel(*args, **kwargs)

    def __repr__(self):
        bound = self._kernel.__name__ if self._kernel else "eager"
        return f"OpSlot({self.op_name}/{self.variant} -> {bound})"
```

### 5. Config → OpSlot Binding Flow

`OpSlot` instances are **module-level globals** in the generated modeling file.
They are created when the module is first imported and start unbound. The user's
config values are passed down during `build_foundation_model`:

```
User YAML                    OpsImplementationConfig              OpSlot.bind()
─────────                    ───────────────────────              ─────────────
model:                       @dataclass
  ops_implementation:  ───→  class OpsImplementationConfig:
    moe_implementation:          moe_implementation: Literal[...]
      fused                      fused_moe_kernel:   Literal[...]
    fused_moe_kernel: triton             │
                                           ▼
                             build_foundation_model(config)
                               ├─ import patched_modeling_qwen3_5_moe_gpu
                               │    └─ module-level OpSlot instances created:
                               │         veomni_apply_rotary_pos_emb = OpSlot(...)
                               │         veomni_moe_experts_forward  = OpSlot(...)
                               │         veomni_load_balancing_loss   = OpSlot(...)
                               │    (all start with _kernel = None)
                               │
                               ├─ _bind_veomni_ops(module, ops_config):
                               │    for name, obj in vars(module).items():
                               │        if isinstance(obj, OpSlot):
                               │            impl = ops_config.resolve_impl_name(obj.op_name)
                               │            obj.bind(impl)     ← resolves via KERNEL_REGISTRY
                               │
                               └─ model init + weight loading
```

After binding, the `OpSlot` globals are live. When methods call
`veomni_moe_experts_forward.has_kernel`, they reference the same bound
module-level object. No additional plumbing is needed — Python's normal
module-global lookup does the work.

```python
# veomni/models/auto.py

def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig):
    """Find all OpSlot instances in the module and bind them."""
    for name, obj in vars(modeling_module).items():
        if isinstance(obj, OpSlot):
            impl_name = ops_config.resolve_impl_name(obj.op_name)
            obj.bind(impl_name)  # validates variant + hardware
```

### 6. Patchgen Integration — Two Options

Both options use `veomni/patchgen/codegen.py`. The generated file is a
self-contained copy of the HF modeling file. The key question is how the
kernel dispatch is wired in.

#### Option A: If-Else Guard (Recommended)

The only diff from upstream HF is a 2-line early-return guard at the top of
each replaceable function or method. Call sites stay unchanged.

**How it works:**

1. Patchgen emits the original HF function/class body verbatim.
2. Patchgen adds `OpSlot` instances at module level (one per replaceable op).
3. Patchgen adds a 2-line guard at the **top of each replaceable function or
   method**:
   ```python
   if veomni_<op>.has_kernel:
       return veomni_<op>(...)
   # original HF code below, unchanged
   ```
4. Call sites (e.g., `Attention.forward` calling `apply_rotary_pos_emb(...)`)
   remain **identical to upstream HF** — no changes needed.
5. At `build_foundation_model` time, `.bind(impl_name)` is called on each
   slot.

**Concrete example — `apply_rotary_pos_emb` in Qwen3.5 MoE:**

Patchgen config:

```python
# qwen3_5_moe_gpu_patch_gen_config.py

config.add_import("veomni.ops.dispatch", names=["OpSlot"])

# Declare op slots
config.add_post_import_block("""
veomni_apply_rotary_pos_emb = OpSlot("apply_rotary_pos_emb", "partial")
""")

# Add guard at top of function — body is original HF code
@config.replace_function("apply_rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # +++ veomni: kernel dispatch +++
    if veomni_apply_rotary_pos_emb.has_kernel:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
    # --- original HF code below, unchanged ---
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

# Attention.forward: NO PATCH — call site stays as HF original:
#   query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

Generated output:

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py

from veomni.ops.dispatch import OpSlot

veomni_apply_rotary_pos_emb = OpSlot("apply_rotary_pos_emb", "partial")

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # +++ veomni: kernel dispatch (2 lines added) +++
    if veomni_apply_rotary_pos_emb.has_kernel:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
    # --- original HF code below, unchanged ---
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

class Qwen3_5MoeAttention(nn.Module):
    ...
    def forward(self, ...):
        ...
        # UNCHANGED from upstream HF — no patching needed at call site
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        ...
```

The diff from upstream HF: **2 lines** added at the top of the function.
The function body is unchanged. All call sites are unchanged.

**Concrete example — `Qwen3_5MoeExperts.forward`:**

Same pattern — early-return guard at the top of the method.

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py

veomni_moe_experts_forward = OpSlot("moe_experts", "standard")

class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... identical to upstream HF ...

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # +++ veomni: kernel dispatch (2 lines added) +++
        if veomni_moe_experts_forward.has_kernel:
            return veomni_moe_experts_forward(
                self, hidden_states, top_k_index, top_k_weights
            )
        # --- original HF code below, unchanged ---
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()
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
                current_hidden_states
                * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states
```

The diff from upstream HF is **2 lines** at the top of `forward`. Everything
below the guard is the original HF code, verbatim.

**Concrete example — `Qwen3_5MoeRMSNorm` (no dispatch needed):**

```python
# In generated/patched_modeling_qwen3_5_moe_gpu.py
# Qwen3.5 MoE uses qwen3_5 variant; no non-eager kernel exists.
# Original HF code emitted verbatim. No slot, no guard.

class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        # ... identical to upstream ...
```

**Properties:**

- **Minimal diff from HF.** 2-line guard at the top of each replaceable
  function/method. Function bodies unchanged. Call sites unchanged.
- **Easy to read.** The guard is self-explanatory: "if veomni has a kernel,
  use it and return; otherwise, fall through to the original HF code below."
- **Easy to diff against upstream.** When HF updates a function body, the
  merge is trivial — only the 2 guard lines at the top are ours.
- Grep `veomni_` to find all dispatch points.
- `repr()` shows what is bound: `OpSlot(moe_experts/standard -> triton_group_gemm)`.

#### Option B: Annotation + Runtime Forward Replacement

Instead of if-else guards, mark replaceable points with decorators. At
`build_foundation_model` time, the resolved kernel is patched onto the
module/class via `setattr`.

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

**Runtime replacement** in `build_foundation_model`:

```python
def _apply_kernel_replacements(modeling_module, ops_config):
    """Walk module namespace, find @veomni_replaceable markers, replace."""
    for name, obj in vars(modeling_module).items():
        if callable(obj) and hasattr(obj, "_veomni_op_name"):
            impl_name = getattr(ops_config, f"{obj._veomni_op_name}_implementation", "eager")
            kernel = KERNEL_REGISTRY.resolve(obj._veomni_op_name, obj._veomni_variant, impl_name)
            if kernel is not None:
                setattr(modeling_module, name, kernel)
    # Similar walk for class methods...
```

**Properties:**

- Zero diff in function bodies — only a decorator line is added.
- Harder to debug: reading the generated code doesn't tell you what runs.
- Module-level `setattr` affects all instances in the process.
- Class replacement needs matching `__init__` signature and parameter names
  for checkpoint loading to work.

### 7. Option Comparison

| Aspect | Option A: If-Else Guard | Option B: Annotation + Replace |
|--------|------------------------|-------------------------------|
| **Diff from HF** | 2-line guard at top of function; body + call sites unchanged | 1-line decorator; body + call sites unchanged |
| **Readability** | Guard is inline and self-explanatory | Must trace build-time `setattr` to understand runtime behavior |
| **Debugging** | Read the code — eager path is right there below the guard | Must inspect module namespace after build |
| **Upstream merges** | Function bodies merge cleanly; only guard lines conflict | Decorator line may conflict; bodies merge cleanly |
| **Internal extension** | Works — registry provides the callable | Works — same registry |
| **Global side effects** | Guard is per-call, no namespace mutation | Module-level `setattr` affects all instances |
| **`torch.compile`** | Simple branch — graph-safe | Function swap before compile — ok if ordered correctly |

**Recommendation: Option A.** The if-else guard makes every dispatch point
visible by reading the generated code. The original HF code is right there
below the guard — no renaming, no wrapping, no hidden mutation. Upstream merges
only touch the guard lines, not the function bodies.

### 8. Loss and MoE LB Coverage

Both loss ops use the same `OpSlot` + if-else guard pattern.

#### Cross-Entropy Loss

```python
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
veomni_seq_cls_loss   = OpSlot("cross_entropy_loss", "seq_cls")

# In ForCausalLM.forward / ForConditionalGeneration.forward:
    if labels is not None:
        # +++ veomni: kernel dispatch +++
        if veomni_causal_lm_loss.has_kernel:
            loss, logits = veomni_causal_lm_loss(logits, labels, self.config)
        else:
            loss = self.loss_function(logits, labels, self.vocab_size, ...)

# In ForSequenceClassification.forward the seq_cls OpSlot is used instead.
```

#### MoE Load-Balancing Loss

```python
veomni_load_balancing_loss = OpSlot("moe_load_balancing_loss", "standard")

# Original HF function — UNCHANGED
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    ...

# Call site in ForCausalLM/ForConditionalGeneration.forward:
    # +++ veomni: kernel dispatch +++
    if veomni_load_balancing_loss.has_kernel:
        aux_loss = veomni_load_balancing_loss(outputs.router_logits, ...)
    else:
        aux_loss = load_balancing_loss_func(outputs.router_logits, ...)
```

### 9. Full Example: Qwen3.5 MoE

**Summary of generated code changes from upstream HF:**

| Component | Change from HF | Lines added |
|-----------|---------------|-------------|
| `apply_rotary_pos_emb` | 2-line guard at top of function | +2 |
| `Qwen3_5MoeExperts.forward` | 2-line guard at top of method | +2 |
| `load_balancing_loss_func` | 2-line guard at top of function | +2 |
| `ForConditionalGeneration.forward` | If-else guard around loss call | +3 |
| `Qwen3_5MoeRMSNorm` | **Unchanged** (qwen3_5 variant, no kernel) | 0 |
| `Qwen3_5MoeAttention.forward` | **Unchanged** (call sites not modified) | 0 |
| Module-level `OpSlot` declarations | New lines | +4 |

Total diff from upstream HF: ~13 lines added. Function bodies unchanged
(only 2-line guards prepended). All call sites unchanged.

**User YAML:**

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    rms_norm_implementation: eager              # only eager for qwen3_5 variant
    rotary_pos_emb_implementation: eager  # only eager for partial variant
    moe_implementation: fused
    fused_moe_kernel: triton
    cross_entropy_loss_implementation: liger_fused
    moe_load_balancing_loss_implementation: eager
```

**Validation at `build_foundation_model` time:**

```
# User mistakenly sets:
rms_norm_implementation: liger

# Error (from KERNEL_REGISTRY.resolve):
KeyError: No kernel 'liger' for op='rms_norm', variant='qwen3_5'.
Available: ['eager']
```

### 10. Lifecycle

```
import veomni                                     # (1) import time
  └─ import kernel_defaults                        #     register OSS kernels

(optional) import internal_kernels.register        # (2) internal registration
  └─ KERNEL_REGISTRY.register(...)                 #     add internal kernels

OpsImplementationConfig.__post_init__()            # (3) config parse time
  └─ rewrite attn_implementation for SP
  └─ expand preset if set

build_foundation_model(config)                     # (4) model build time
  ├─ import patched_modeling_qwen3_5_moe_gpu       #     generated module
  │    └─ module-level OpSlot instances created:    #     (at import time)
  │         veomni_apply_rotary_pos_emb  = OpSlot("apply_rotary_pos_emb", "partial")
  │         veomni_moe_experts_forward   = OpSlot("moe_experts", "standard")
  │         veomni_load_balancing_loss   = OpSlot("moe_load_balancing_loss", "standard")
  │         veomni_causal_lm_loss        = OpSlot("cross_entropy_loss", "causal")
  │         (all start with _kernel = None)
  │
  ├─ _bind_veomni_ops(module, ops_config):          #     bind from config
  │    for each OpSlot in vars(module):
  │        impl = ops_config.resolve_impl_name(slot.op_name)  # e.g. "triton"
  │        slot.bind(impl)                                    # KERNEL_REGISTRY.resolve()
  │
  └─ model init + weight loading

model.forward()                                    # (5) runtime
  ├─ apply_rotary_pos_emb(...)                     #     function called as normal
  │   └─ if veomni_apply_rotary_pos_emb.has_kernel:#     guard checks module global
  │        return veomni_apply_rotary_pos_emb(...)  #     → fused kernel
  │      else: <original HF code>                   #     → eager fallback
  ├─ attention: ALL_ATTENTION_FUNCTIONS[...]         #     unchanged
  ├─ experts.forward(...)
  │   └─ if veomni_moe_experts_forward.has_kernel:
  │        return veomni_moe_experts_forward(...)    #     → fused kernel
  │      else: <original HF expert loop>             #     → eager fallback
  ├─ if veomni_causal_lm_loss.has_kernel: ...        #     guard in forward
  └─ if veomni_load_balancing_loss.has_kernel: ...   #     guard in forward
```

---

## Migration

| Current mechanism | New mechanism | Migration |
|---|---|---|
| `VEOMNI_USE_LIGER_KERNEL=1` env var | `rms_norm_implementation: liger` etc. | Deprecate env var; keep compat for 1 release |
| `gpu_patch.py` monkey-patching | patchgen + `OpSlot` guards | Remove `gpu_patch.py` files |
| `apply_veomni_loss_patch()` at import | `cross_entropy_loss_implementation` + `OpSlot` | Remove import-time patch |
| `apply_veomni_fused_moe_patch()` | `OpSlot("moe_experts", ...)` | Kept for legacy-dispatch models (qwen3_moe etc.) until they adopt OpSlot |
| `moe_implementation: fused_quack` (single field) | `moe_implementation=fused` + `fused_moe_kernel=quack` | Legacy value accepted with DeprecationWarning |

---

## Open Questions

1. **Preset system:** Should `preset: liger` silently skip ops where the
   model's variant has no `liger` registration, or warn?
2. **Attention:** Keep in `ALL_ATTENTION_FUNCTIONS` (shared with HF) or unify
   under `KERNEL_REGISTRY`?
3. **NPU auto-selection:** Should NPU be an explicit `npu_group_gemm`
   implementation name, or remain automatic?
4. **Multi-model processes:** Each generated module has its own `OpSlot`
   instances, so different models work independently. But two instances of the
   same model with different ops configs share the same `OpSlot` objects.
   Is this a real use case?
