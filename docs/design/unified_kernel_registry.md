# Unified Kernel Registry

## Status

**Historical proposal** | 2026-03-16

> This document records the original registry design and contains illustrative
> pseudocode. It is not the source of truth for runnable configuration. See
> [Kernel Selection in VeOmni](kernel_selection.md) for the current config
> fields, defaults, backends, lifecycle, and implementation paths.

## Problem

At the time of this proposal, VeOmni's kernel selection was fragmented:

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
| `moe_experts` | `standard` | Expert GEMM dispatch (merged gate+up projection, HF v5 convention; DeepSeek-V4 additionally forwards `swiglu_limit` to clamp-aware backends) |
| `cross_entropy_loss` | `causal`, `seq_cls` | Causal-LM CE (shifts labels) vs. sequence-classification CE (no shift) |
| `load_balancing_loss` | `standard` | Switch Transformer auxiliary loss |

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

    def register(self, spec: KernelSpec, force=False) -> None:
        key = (spec.op_name, spec.variant)
        bucket = self._specs.setdefault(key, {})
        if spec.name in bucket:
            if force:
                logger.info(
                    f"Kernel(op='{spec.op_name}', variant='{spec.variant}', name='{spec.name}') is replaced with a new one from {spec.factory.__code__.co_filename}"
                )
            else:
                raise ValueError(
                    f"Duplicate kernel registration: op='{spec.op_name}', variant='{spec.variant}', name='{spec.name}'"
                )
        bucket[spec.name] = spec

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

**Illustrative proposal registrations.** The shipped registrations are
distributed under `veomni/ops/kernels/*` and `veomni/ops/liger/`; they are
imported by `veomni/ops/__init__.py`.

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
# NPU chunk-loss backs the causal variant only; chunk_loss hard-codes the
# `labels[..., 1:]` shift so ForSequenceClassification stays on eager.
KERNEL_REGISTRY.register(KernelSpec(
    name="npu",
    op_name="cross_entropy_loss", variant="causal",
    factory=_npu_chunk_loss_causal_factory,   # chunk_loss_function (handles SP reduction internally)
    hardware=HardwareRequirement("npu"),
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

    # Per-op fields remain strings so third-party backends can register values.
    moe_implementation: str = "fused_triton"
    cross_entropy_loss_implementation: str = "liger_kernel"
    rms_norm_implementation: str = "liger_kernel"
    swiglu_mlp_implementation: str = "liger_kernel"
    rotary_pos_emb_implementation: str = "liger_kernel"
    rotary_pos_emb_vision_implementation: str = "eager"
    load_balancing_loss_implementation: str = "triton"
    rms_norm_gated_implementation: str = "fla"
    causal_conv1d_implementation: str = "fla"
    chunk_gated_delta_rule_implementation: str = "fla"
    dsa_indexer_implementation: Literal["eager", "cudnn", "tilelang"] = "eager"
    dsa_attention_implementation: Literal["eager", "flashmla_cudnn", "tilelang"] = "eager"
    mhc_implementation: Literal["eager", "tilelang"] = "eager"
```

**Shipped today** (what is actually on `OpsImplementationConfig` as of this
PR — see `veomni/arguments/arguments_types.py`):

| Field | Available values | Notes |
|-------|------------------|-------|
| `attn_implementation` | `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `native-sparse` | VeOmni rewrites FA2/3/4 to SP-aware variants under `MODELING_BACKEND=veomni` |
| `rms_norm_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3) | |
| `rotary_pos_emb_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3) | |
| `swiglu_mlp_implementation` | `eager`, `liger_kernel` | |
| `moe_implementation` | `eager`, `fused_triton`, `fused_quack`, `fused_npu` | Single field. On NPU, a value still equal to the GPU default `fused_triton` is normalized to `fused_npu`; incompatible non-default overrides raise. |
| `cross_entropy_loss_implementation` | `eager`, `liger_kernel`, `chunk_loss`, `npu` | |
| `load_balancing_loss_implementation` | `eager`, `triton` | `triton` is CUDA-only; current NPU config normalization maps the default-valued `triton` selection to `eager` before binding. |
| `rms_norm_gated_implementation` | `eager`, `fla`, `npu` | Qwen3.5 GatedDeltaNet `self.norm`; default `fla` |
| `causal_conv1d_implementation` | `eager`, `fla`, `npu` | Qwen3.5 GatedDeltaNet pre-mixer; `eager` has no `cu_seqlens` path |
| `chunk_gated_delta_rule_implementation` | `eager`, `fla`, `flash_qla`, `npu` | Qwen3.5 linear attention; `flash_qla` is Hopper SM90-only, while `npu` uses the vendored MindSpeed-MM kernel |
| `dsa_indexer_implementation` | `eager`, `cudnn`, `tilelang` | GLM-DSA supports `cudnn`; DeepSeek V4 supports `tilelang`. Optimized implementations require compatible NVIDIA hardware. |
| `dsa_attention_implementation` | `eager`, `flashmla_cudnn`, `tilelang` | GLM-DSA supports `flashmla_cudnn`; DeepSeek V4 supports `tilelang`. Optimized implementations require compatible NVIDIA hardware. |
| `mhc_implementation` | `eager`, `tilelang` | DeepSeek V4 manifold-constrained Hyper-Connection pre/post/head kernels provided by the `tile-kernels` package. The three `OpSlot("mhc", variant)` instances share this selection and require NVIDIA SM90+ for `tilelang`. |

No preset field was shipped. Configure each `OpsImplementationConfig` field
explicitly; unknown fields such as `preset` are invalid.

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
    def use_non_eager_impl(self) -> bool:
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
    moe_implementation:          moe_implementation: str
      fused_triton                         │
                                           ▼
                             build_foundation_model(config, ops_implementation=ops)
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
                               │            impl = getattr(ops_config, f"{obj.op_name}_implementation", "eager")
                               │            obj.bind(impl)     ← resolves via KERNEL_REGISTRY
                               │
                               └─ model init + weight loading
```

After binding, the `OpSlot` globals are live. When methods call
`veomni_moe_experts_forward.use_non_eager_impl`, they reference the same bound
module-level object. No additional plumbing is needed — Python's normal
module-global lookup does the work.

```python
# veomni/models/auto.py

def _bind_veomni_ops(modeling_module, ops_config: OpsImplementationConfig):
    """Find all OpSlot instances in the module and bind them."""
    for name, obj in vars(modeling_module).items():
        if isinstance(obj, OpSlot):
            # `moe_experts` is the one op whose config field is `moe_implementation`
            # (not `moe_experts_implementation`) and whose values carry a `fused_`
            # prefix that registry entries don't — translate here.
            if obj.op_name == "moe_experts":
                impl_name = (
                    "eager" if ops_config.moe_implementation == "eager"
                    else ops_config.moe_implementation.removeprefix("fused_")
                )
            else:
                impl_name = getattr(ops_config, f"{obj.op_name}_implementation", "eager")
            obj.bind(impl_name)  # validates variant + hardware
```

### 6. Patchgen Integration — Two Options

Both options use `veomni/patchgen/codegen.py`. The generated file is a
self-contained copy of the HF modeling file. The key question is how the
kernel dispatch is wired in.

> **Proposal-only examples:** The Qwen3.5 MoE snippets in this section record
> alternatives considered in March 2026; they are not excerpts from the
> current generated GPU module. Current Qwen3.5 MoE has no rotary `OpSlot` and
> exposes the seven slots listed under
> [Current generated dispatch coverage](#current-generated-dispatch-coverage).

#### Option A: If-Else Guard (Recommended)

The only diff from upstream HF is a 2-line early-return guard at the top of
each replaceable function or method. Call sites stay unchanged.

**How it works:**

1. Patchgen emits the original HF function/class body verbatim.
2. Patchgen adds `OpSlot` instances at module level (one per replaceable op).
3. Patchgen adds a 2-line guard at the **top of each replaceable function or
   method**:
   ```python
   if veomni_<op>.use_non_eager_impl:
       return veomni_<op>(...)
   # original HF code below, unchanged
   ```
4. Call sites (e.g., `Attention.forward` calling `apply_rotary_pos_emb(...)`)
   remain **identical to upstream HF** — no changes needed.
5. At `build_foundation_model` time, `.bind(impl_name)` is called on each
   slot.

**Original proposal example — a partial-RoPE guard:**

Patchgen config:

```python
# Proposal pseudocode; not the current Qwen3.5 MoE GPU patchgen config.

config.add_import("veomni.ops.dispatch", names=["OpSlot"])

# Declare op slots
config.add_post_import_block("""
veomni_apply_rotary_pos_emb = OpSlot("apply_rotary_pos_emb", "partial")
""")

# Add guard at top of function — body is original HF code
@config.replace_function("apply_rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # +++ veomni: kernel dispatch +++
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
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

Illustrative output proposed at the time:

```python
# Proposal pseudocode; not the current generated Qwen3.5 MoE module.

from veomni.ops.dispatch import OpSlot

veomni_apply_rotary_pos_emb = OpSlot("apply_rotary_pos_emb", "partial")

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # +++ veomni: kernel dispatch (2 lines added) +++
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
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

Under this proposal, the diff from upstream HF would have been two guard lines
at the top of the function, with unchanged call sites.

**Original proposal example — `Qwen3_5MoeExperts.forward`:**

Same pattern — early-return guard at the top of the method.

```python
# Proposal pseudocode illustrating the guard shape.

veomni_moe_experts_forward = OpSlot("moe_experts", "standard")

class Qwen3_5MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... identical to upstream HF ...

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # +++ veomni: kernel dispatch (2 lines added) +++
        if veomni_moe_experts_forward.use_non_eager_impl:
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

Under the proposal, the only conceptual addition was an early-return guard;
inspect the current generated module for the exact shipped output.

**Superseded proposal assumption — `Qwen3_5MoeRMSNorm` without dispatch:**

```python
# Historical proposal only. Current Qwen3.5 MoE has
# OpSlot("rms_norm", "qwen3_5"); this no-slot assumption was superseded.

class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        # ... identical to upstream ...
```

The shipped implementation now binds RMSNorm through its `qwen3_5` variant;
this snippet is retained only to explain the rejected alternative.

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
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits = veomni_causal_lm_loss(logits, labels, self.config)
        else:
            loss = self.loss_function(logits, labels, self.vocab_size, ...)

# In ForSequenceClassification.forward the seq_cls OpSlot is used instead.
```

#### MoE Load-Balancing Loss

```python
veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")

# Original HF function — UNCHANGED
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0
    ...

# Call site in ForCausalLM/ForConditionalGeneration.forward:
    # +++ veomni: kernel dispatch +++
    if veomni_load_balancing_loss.use_non_eager_impl:
        aux_loss = veomni_load_balancing_loss(outputs.router_logits, ...)
    else:
        aux_loss = load_balancing_loss_func(outputs.router_logits, ...)
```

### 9. Full Example: Qwen3.5 MoE

#### Current generated dispatch coverage

| OpSlot | Purpose |
|--------|---------|
| `OpSlot("rms_norm", "qwen3_5")` | Qwen3.5 MoE RMSNorm |
| `OpSlot("moe_experts", "standard")` | Fused or eager MoE expert forward |
| `OpSlot("cross_entropy_loss", "causal")` | Causal language-model loss |
| `OpSlot("load_balancing_loss", "standard")` | MoE auxiliary load-balancing loss |
| `OpSlot("rms_norm_gated", "standard")` | GatedDeltaNet gated RMSNorm |
| `OpSlot("causal_conv1d", "standard")` | GatedDeltaNet causal convolution |
| `OpSlot("chunk_gated_delta_rule", "standard")` | GatedDeltaNet recurrent update |

Patchgen injects these module-level slots and narrow guards at their current
call sites. The exact generated line count is intentionally not treated as an
API guarantee and may change when the upstream Transformers implementation
changes.

**User YAML:**

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    rms_norm_implementation: liger_kernel
    rotary_pos_emb_implementation: eager
    moe_implementation: fused_triton
    cross_entropy_loss_implementation: liger_kernel
    load_balancing_loss_implementation: eager
```

**Validation at `build_foundation_model` time:**

```
# User mistakenly sets:
rms_norm_implementation: unknown_backend

# Error (from KERNEL_REGISTRY.resolve):
KeyError: Unknown kernel 'unknown_backend' for op='rms_norm', variant='qwen3_5'.
```

### 10. Lifecycle

```
import veomni                                     # (1) import time
  └─ import veomni.ops.{kernels,liger}             #     register OSS kernels

(optional) import internal_kernels.register        # (2) internal registration
  └─ KERNEL_REGISTRY.register(...)                 #     add internal kernels

OpsImplementationConfig.__post_init__()            # (3) config parse time
  └─ rewrite attn_implementation for SP
  └─ validate configured backends

build_foundation_model(config, ops_implementation=ops) # (4) model build time
  ├─ import patched_modeling_qwen3_5_moe_gpu       #     generated module
  │    └─ module-level OpSlot instances created:    #     (at import time)
  │         veomni_rms_norm               = OpSlot("rms_norm", "qwen3_5")
  │         veomni_moe_experts_forward   = OpSlot("moe_experts", "standard")
  │         veomni_load_balancing_loss   = OpSlot("load_balancing_loss", "standard")
  │         veomni_causal_lm_loss        = OpSlot("cross_entropy_loss", "causal")
  │         veomni_rms_norm_gated        = OpSlot("rms_norm_gated", "standard")
  │         veomni_causal_conv1d         = OpSlot("causal_conv1d", "standard")
  │         veomni_chunk_gated_delta_rule= OpSlot("chunk_gated_delta_rule", "standard")
  │         (all start with _kernel = None)
  │
  ├─ _bind_veomni_ops(module, ops_config):          #     bind from config
  │    for each OpSlot in vars(module):
  │        impl = ops_config.moe_implementation.removeprefix("fused_")   # if moe_experts
  │              or getattr(ops_config, f"{slot.op_name}_implementation", "eager")
  │        slot.bind(impl)                                    # KERNEL_REGISTRY.resolve()
  │
  └─ model init + weight loading

model.forward()                                    # (5) runtime
  ├─ RMSNorm / gated RMSNorm / causal Conv1D / gated delta rule
  │   └─ each guard dispatches through its bound OpSlot or keeps the HF path
  ├─ attention: ALL_ATTENTION_FUNCTIONS[...]        #     separate attention dispatch
  ├─ experts.forward(...)
  │   └─ if veomni_moe_experts_forward.use_non_eager_impl:
  │        return veomni_moe_experts_forward(...)    #     → fused kernel
  │      else: <original HF expert loop>             #     → eager fallback
  ├─ if veomni_causal_lm_loss.use_non_eager_impl: ...        #     guard in forward
  └─ if veomni_load_balancing_loss.use_non_eager_impl: ...   #     guard in forward
```

---

## Migration

| Current mechanism | New mechanism | Migration |
|---|---|---|
| `VEOMNI_USE_LIGER_KERNEL=1` env var | Per-op `*_implementation: liger_kernel` fields | Removed; configure fields explicitly |
| `gpu_patch.py` monkey-patching | patchgen + registry/`OpSlot` dispatch | Removed from current model paths |
| `apply_veomni_loss_patch()` at import | `cross_entropy_loss_implementation` + `apply_ops_config()` | Replaced by the unified config install point |
| `apply_veomni_fused_moe_patch()` | `OpSlot("moe_experts", ...)` | All MoE models (qwen3_moe, qwen3_5_moe, qwen3_vl_moe, qwen3_omni_moe, deepseek_v3, deepseek_v4) now bind through OpSlot guards; the function is kept only as the binding helper invoked from `_bind_veomni_ops` to set the global `_fused_moe_forward` pointer. DeepSeek-V4 separately exposes TileLang DSA indexer/attention config slots and three registry-backed TileKernels mHC slots. Its MoE keeps a direct `fused_moe_forward(...)` call under the experts guard so it can pass its merged `gate_up_proj` layout and `swiglu_limit` clamp explicitly; clamp-aware V4 fused MoE is currently provided by the GPU backends and defaults to `fused_triton` on GPU, while `fused_npu` raises until the NPU kernel implements `swiglu_limit`. |
| `moe_implementation: fused` | `moe_implementation: fused_triton`, `fused_quack`, or `fused_npu` | The legacy `"fused"` alias remains deprecated: it resolves to `fused_quack` on GPU and `fused_npu` on NPU with a warning. The default-valued `fused_triton` selection is also normalized to `fused_npu` on NPU for compatibility; explicit backend names are recommended. |

---

## Open Questions

1. ~~**Preset system:** Should a Liger preset silently skip unsupported
   ops?~~ **Resolved:** no preset field was shipped; users configure each op.
2. **Attention:** Keep in `ALL_ATTENTION_FUNCTIONS` (shared with HF) or unify
   under `KERNEL_REGISTRY`?
3. ~~**NPU auto-selection:** Should NPU be an explicit `npu_group_gemm`
   implementation name, or remain automatic?~~ **Resolved:**
   `moe_implementation: fused_npu` is the explicit, recommended spelling. For
   compatibility, the GPU dataclass default `fused_triton` still normalizes to
   `fused_npu` on NPU; incompatible non-default overrides raise.
4. **Multi-model processes:** Each generated module has its own `OpSlot`
   instances, so different models work independently. But two instances of the
   same model with different ops configs share the same `OpSlot` objects.
   Is this a real use case?
