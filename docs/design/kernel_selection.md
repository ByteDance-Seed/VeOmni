# Kernel Selection in VeOmni

VeOmni selects optimized kernel implementations for attention, mHC,
cross-entropy loss, Liger fused ops (RMSNorm, RoPE, SwiGLU), MoE, and
load-balancing loss.
All selections are driven by config fields in `OpsImplementationConfig`.

## Quick Reference

Every configurable kernel lives under `model.ops_implementation.*` in YAML and
maps to a field on `OpsImplementationConfig` (`veomni/arguments/arguments_types.py`).
Below is the full list — if a field is not in this table, it is not a kernel
selection knob.

| Kernel | Config field | Available values | Default | Selection time |
|--------|-------------|------------------|---------|----------------|
| Attention | `attn_implementation` | `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`, `flex_attention`, `native-sparse` | `"flash_attention_2"` | Config `__post_init__` + `build_foundation_model` |
| DSA indexer | `dsa_indexer_implementation` | `eager`, `cudnn` (GLM-DSA), `tilelang` (DeepSeek-V4) | `"eager"` | Model `__init__` via an instance-local `VeomniKernel` |
| DSA attention | `dsa_attention_implementation` | `eager`, `flashmla_cudnn` (GLM-DSA), `tilelang` (DeepSeek-V4) | `"eager"` | Model `__init__` via an instance-local `VeomniKernel` |
| mHC | `mhc_implementation` | `eager`, `tilelang` (DeepSeek-V4, SM90+) | `"eager"` | Model `__init__` via three instance-local `VeomniKernel`s (`pre`, `post`, `head`) |
| Cross-entropy loss | `cross_entropy_loss_implementation` | `eager`, `liger_kernel`, `chunk_loss`, `npu` | `"liger_kernel"` (GPU) | Model `__init__` via an instance-local `VeomniKernel` |
| RMSNorm | `rms_norm_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3) | `"liger_kernel"` (GPU) | Model `__init__` via an instance-local `VeomniKernel` |
| SwiGLU MLP | `swiglu_mlp_implementation` | `eager`, `liger_kernel` | `"liger_kernel"` (GPU) | Model `__init__` via an instance-local `VeomniKernel` |
| Rotary embedding | `rotary_pos_emb_implementation` | `eager`, `liger_kernel`, `npu`, `triton` (per-model; DeepSeek-V3, DeepSeek-V4, Wan) | `"liger_kernel"` (GPU) | Model `__init__` via an instance-local `VeomniKernel` |
| Vision rotary embedding | `rotary_pos_emb_vision_implementation` | `eager`, `npu` | `"eager"` | Model `__init__` via an instance-local `VeomniKernel` |
| Gated RMSNorm | `rms_norm_gated_implementation` | `eager`, `fla`, `npu` | `"fla"` (GPU) | Qwen3.5 model `__init__` via an instance-local `VeomniKernel` |
| Causal Conv1D | `causal_conv1d_implementation` | `eager`, `fla`, `npu` | `"fla"` (GPU) | Qwen3.5 model `__init__` via an instance-local `VeomniKernel` |
| Gated delta rule | `chunk_gated_delta_rule_implementation` | `eager`, `fla`, `flash_qla` (SM90), `npu`, `npu_ascendc` | `"fla"` (GPU) | Qwen3.5 model `__init__` via an instance-local `VeomniKernel` |
| Load-balancing loss | `load_balancing_loss_implementation` | `eager`, `triton` (CUDA; NPU config normalizes this default to `eager`) | `"triton"` | Model `__init__` via an instance-local `VeomniKernel` |
| MoE experts | `moe_implementation` | `eager`, `triton`, `quack` (SM90+), `npu`, `mlu` | `"triton"` (GPU) | Model `__init__` via an instance-local `VeomniKernel` |

**Most optimized-op defaults are GPU-oriented.** On Ascend NPU, values still
equal to the dataclass defaults automatically resolve to `npu` for RMSNorm,
rotary embedding, vision rotary embedding, and cross-entropy; to `npu`
for MoE; and to `eager` for SwiGLU and load-balancing loss. Explicit
non-default overrides are retained and rejected when unsupported. Qwen3.5's
three GatedDeltaNet fields are
model-specific and are not auto-resolved: set them to `npu` explicitly because
the causal-convolution and gated-delta-rule `eager` fallbacks do not support
dynamic-batch `cu_seqlens`.

The general per-op fields are typed as plain `str` (not `Literal`), so
third-party backends can call `register_kernel(...)` without modifying
`OpsImplementationConfig`.

---

## Lifecycle Overview

```
import veomni                                 # (1) import time
  └─ import veomni.kernels
       └─ apply_kernel_patch()
            └─ apply_veomni_attention_patch() # register Flash/Flex facade names with SP

OpsImplementationConfig.__post_init__()       # (2) config parse time
  ├─ validate requested backends are available
  └─ rewrite attn_implementation for SP

BaseTrainer._build_model()                    # (3) model build time
  └─ models_kernel.build_foundation_model(..., kernels_implementation=ops)
       ├─ set_kernels_config(ops)
       └─ model init + weight loading
            ├─ patched modules construct their local VeomniKernel handles
            ├─ self.veomni_ce = VeomniKernel("cross_entropy_loss", ...)
            ├─ self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
            └─ MoE models bind experts and load-balancing helpers to local handles

model.forward()                               # (4) runtime
  ├─ attention: ALL_ATTENTION_FUNCTIONS[config._attn_implementation]
  ├─ loss: self.loss_function(...) -> model helper -> instance-local VeomniKernel
  ├─ RMSNorm/RoPE/SwiGLU: Liger or HF default (set at registration)
  ├─ mHC: TileKernels pre/post/head or original Transformers implementation
  └─ MoE: fused_moe_forward(...) or eager loop
```

**No global loss mutation.** The current `models_kernel` stack does not replace
Transformers' `LOSS_MAPPING`. Each model resolves its configured CE row once,
stores an instance-local `VeomniKernel`, and binds that handle to the model
helper with `functools.partial`; there is no per-forward implementation lookup.

**Ownership.** `models_kernel.build_foundation_model` installs the supplied
`kernels_implementation` through `set_kernels_config` before constructing the
model. Callers that provide neither an explicit config nor a previously
installed kernel config receive `ValueError`; there is no silent all-eager
fallback.

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
| `flex_attention` | PyTorch FlexAttention | Yes | Native `BlockMask`; CUDA for compiled training |
| `native-sparse` | Sparse attention | No | — |

When `MODELING_BACKEND=veomni` (the default), `__post_init__` automatically
rewrites `flash_attention_2/3/4` and `flex_attention` to VeOmni SP-aware
variants (`veomni_flash_attention_*` and `veomni_flex_attention`). These
wrappers apply the shared Ulysses helpers before dispatching to the selected
backend.

FlexAttention requires a model-provided native `BlockMask`; VeOmni does not
construct model-specific visibility. With Ulysses enabled, the mask must be
head-broadcast (`BlockMask.shape[1] == 1`) because rank-local head indices are
not rebased for head-specific masks. See
`docs/transformers_v5/veomni_fused_attention.md` for the full contract.

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Installation: `veomni/kernels/install.py` — `apply_kernel_patch()` / `apply_veomni_attention_patch()`
- Plumbing: `veomni/models_kernel/auto.py` — `build_foundation_model(kernels_implementation=...)`

### DeepSeek V4 DSA and mHC

DeepSeek V4 exposes three independent model-specific selections. The TileLang
indexer and sparse attention backends support packed dynamic batches; the
TileKernels mHC backend replaces the pre/Sinkhorn/collapse, residual post-mix,
and final head collapse through three instance-local `VeomniKernel` handles:

```yaml
model:
  ops_implementation:
    dsa_indexer_implementation: tilelang
    dsa_attention_implementation: tilelang
    mhc_implementation: tilelang
```

All three optimized paths require NVIDIA SM90 or later. `mhc_implementation` defaults
to `eager` and never silently falls back after `tilelang` is selected. The mHC
implementation is provided by the `tile-kernels` package.
TileKernels' training path supports forward and backward with BF16 activations
and DeepSeek V4's `hc_mult=4` layout.

---

## 2. Cross-Entropy Loss

### Config

```yaml
model:
  ops_implementation:
    cross_entropy_loss_implementation: liger_kernel   # default; set to "chunk_loss" / "npu" / "eager" on NPU
```

**Field:** `OpsImplementationConfig.cross_entropy_loss_implementation`

### Available implementations

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | Fused linear + CE raw forward/backward | `liger-kernel` package + CUDA |
| `chunk_loss` | Chunked `F.linear` + CE with one global denominator | — |
| `npu` | Config alias resolved to `chunk_loss` by model construction | Ascend NPU config |
| `eager` | PyTorch `F.linear`/`F.cross_entropy` | — |

All registered rows implement the same token-level `standard` contract:
`(hidden_or_logits, labels, weight, *, ignore_index, num_items_in_batch)`.
An empty weight marks a logits input and is supported by eager; the fused rows
require a projection weight. Causal shifting, sequence-classification label
policy, and SP reduction are outside the kernel in
`models_kernel/loss_utils/cross_entropy_loss.py`, so chunked CE is no longer
causal-only.

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

- Registration: `veomni/kernels/_kernels/loss/__init__.py`
- Implementations: `veomni/kernels/_kernels/loss/cross_entropy_loss/standard/`
- Model policy: `veomni/models_kernel/loss_utils/cross_entropy_loss.py`
- Log-probs/distillation side paths: `veomni/models_kernel/loss_utils/`

---

## 3. Per-Model Ops (RMSNorm, RoPE, SwiGLU MLP)

Each operation can be independently controlled. Despite the historical
"Liger fused ops" label, these fields are *not* Liger-only: they also accept
`npu` (for Ascend NPU backends) and `triton` (for model-specific registry
variants, e.g. DeepSeek-V3's batch-invariant RMSNorm and deterministic RoPE).

### Config

```yaml
model:
  ops_implementation:
    rms_norm_implementation: liger_kernel       # default; pin to "npu" / "eager" on NPU
    swiglu_mlp_implementation: liger_kernel     # default; pin to "eager" on NPU (no NPU backend)
    rotary_pos_emb_implementation: liger_kernel # default; pin to "npu" / "eager" on NPU
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
| `triton` | Model-specific Triton kernel registered via `extra_backends` (DeepSeek-V3 deterministic RoPE, DeepSeek-V4 fused partial-interleaved RoPE, Wan DiT) | `triton`, per-model registration |
| `eager` | HuggingFace default (`apply_rotary_pos_emb`) | — |

#### `swiglu_mlp_implementation`

| Value | Implementation | Requirements |
|-------|---------------|---|
| `liger_kernel` | `LigerSwiGLUMLP` | `liger-kernel` package |
| `eager` | HuggingFace default (`{Model}MLP`) | — |

### What gets patched

For each selected backend, patchgen-generated modeling constructs a
model-local `VeomniKernel` handle and calls it from the relevant method.
Registry wrappers preserve model-specific call contracts and can accept
model-specific arguments such as an optional RMSNorm weight:

| Config field | Original | Liger replacement |
|---|---|---|
| `rms_norm_implementation` | `{Model}RMSNorm` | Functional Liger RMSNorm |
| `rotary_pos_emb_implementation` | `apply_rotary_pos_emb` | `liger_rotary_pos_emb` |
| `swiglu_mlp_implementation` | `{Model}MLP.forward` | Functional Liger SwiGLU |

The `npu` and `triton` backends use the same handle flow; the selected registry
row determines the callable and validates its hardware requirement.

### Models with Liger support

Qwen2, Qwen3, Qwen3-MoE, Qwen2-VL, DeepSeek-V3, DeepSeek-V4, Llama,
Seed-OSS. DeepSeek-V4 supports weighted and unweighted RMSNorm plus a
clamp-preserving Liger silu*mul path for shared experts. Its partial
interleaved RoPE has no Liger equivalent, so `liger_kernel` is rejected for
`rotary_pos_emb_implementation` on that model; the supported values are
`triton` (the fused kernel above) and `eager`.

### Key files

- Config singleton: `veomni/kernels/config.py` — `get_kernels_config()`, `set_kernels_config()`
- Unified registry: `veomni/kernels/registry.py` — `register_kernel()`, `resolve_kernel()`, `VeomniKernel`
- Backend registration: `veomni/kernels/_kernels/{rms_norm,rope,swiglu_mlp}/__init__.py`
- Model integration: `veomni/models_kernel/transformers/{model}/*_patch_gen_config.py`

---

## 4. Qwen3.5 GatedDeltaNet Ops

Qwen3.5 exposes three additional model-specific kernel fields. They default to
the GPU `fla` implementations, so NPU users must select `npu` explicitly:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: npu
    causal_conv1d_implementation: npu
    chunk_gated_delta_rule_implementation: npu
```

| Field | GPU values | NPU value | Eager limitation |
|---|---|---|---|
| `rms_norm_gated_implementation` | `fla` | `npu` | HuggingFace reference implementation |
| `causal_conv1d_implementation` | `fla` | `npu` | No `cu_seqlens` path |
| `chunk_gated_delta_rule_implementation` | `fla`, `flash_qla` (SM90 only) | `npu`, `npu_ascendc` | No `cu_seqlens` path |

The NPU gated RMSNorm uses `torch_npu`. The NPU causal Conv1D and gated
delta-rule implementations additionally require `triton-ascend`. For the gated
delta rule there are two NPU backends: `npu` (the vendored MindSpeed-MM Triton
kernel) and `npu_ascendc` (an AscendC fused `torch.ops.npu.*` path), the latter
requiring a manual `fla_npu` install. Registrations live in
`veomni/kernels/_kernels/gated_delta_rule/__init__.py`; field defaults and allowed
values are documented by `OpsImplementationConfig`.

---

## 5. Load-Balancing Loss

### Config

```yaml
model:
  ops_implementation:
    load_balancing_loss_implementation: triton   # CUDA
    # load_balancing_loss_implementation: eager  # NPU
```

**Field:** `OpsImplementationConfig.load_balancing_loss_implementation`

### Available implementations

| Value | Implementation | Requirements |
|-------|---------------|---|
| `triton` | Fused tensor-native `[N, E]` kernel | `triton` on CUDA |
| `eager` | Pure-PyTorch tensor-native `[N, E]` reference | — |

Normal NPU config construction maps every value equal to the dataclass default
`triton`—including an explicit YAML value—to `eager` before registry binding.
The optimized `triton` implementation is CUDA-only; select `eager` in current
NPU configs.

This loss has no compatibility facade or process-global dispatch. Each MoE model creates
`VeomniKernel("load_balancing_loss", "standard", impl)` in `__init__`, then
binds the model-facing helper with `partial(load_balancing_loss,
kernel=self.veomni_lb)`. The helper preserves the HF-shaped API and converts
the tuple of per-layer router logits into the raw kernel's `[N, E]` input.

### Key files

- Model helper: `veomni/models_kernel/loss_utils/load_balancing_loss.py`
- Triton impl: `veomni/kernels/_kernels/loss/load_balancing_loss/standard/triton.py`
- Eager impl: `veomni/kernels/_kernels/loss/load_balancing_loss/standard/eager.py`
- Registration: `veomni/kernels/_kernels/loss/__init__.py`

---

## 6. MoE Kernel

### Config

```yaml
model:
  ops_implementation:
    moe_implementation: triton   # Triton group-gemm (GPU SM70+ or MLU)
    # moe_implementation: quack   # Quack CUTLASS/CuTe (GPU, SM90+)
    # moe_implementation: npu     # NPU group-gemm (Ascend)
    # moe_implementation: mlu     # Apex grouped-GEMM (MLU)
    # moe_implementation: eager   # Reference PyTorch loop (very slow, debug only)
```

**Field:** `OpsImplementationConfig.moe_implementation`
**Default:** `"triton"` (GPU). On NPU, a value still equal to this
dataclass default—including an explicit YAML value—is normalized to
`"npu"`. Set `"npu"` explicitly for clarity; incompatible
non-default overrides such as `"quack"` raise at config validation time.

The mode and kernel backend are expressed as a single field. After the
default-value compatibility normalization above, remaining hardware mismatches
raise during config validation or kernel binding.

| Value | Kernel | Hardware | EP support |
|-------|--------|----------|:----------:|
| `eager` | PyTorch expert loop | Any | No |
| `triton` | Triton group-gemm | GPU SM70+ or MLU | Yes |
| `quack` | Quack CUTLASS/CuTe | GPU, SM90+ (H100+) | No |
| `npu` | NPU group-gemm | Ascend NPU | Yes |
| `mlu` | Apex grouped-GEMM | Cambricon MLU | Yes |

DeepSeek-V4 keeps eager DSA indexer and attention as its defaults, with optional
SM90+ `tilelang` indexer and attention implementations. Its MoE path
uses the independent `moe_implementation` selection and therefore defaults to
`triton` on GPU.
The v4-specific patched experts path passes the merged `gate_up_proj` tensor
directly to `fused_moe_forward(...)` and forwards `swiglu_limit` so every
supported fused backend preserves V4's clamped SwiGLU pre-activation semantics.
On Ascend, `npu` keeps the existing `torch_npu.npu_swiglu` path when no
limit is configured and uses a forward/backward `triton-ascend` kernel for the
clamped DeepSeek-V4 path when the Ascend backend is available. The import stays lazy, so
other NPU MoE models do not gain a Triton dependency. A bare or legacy NPU
environment preserves the original eager clamp, SiLU, and multiply training
path instead. VeOmni's product-based Ascend images install and verify
`triton-ascend`; other environments can install a release compatible with their
CANN and `torch_npu` stack to enable the fused activation.

### Key files

- Config: `veomni/arguments/arguments_types.py` — `OpsImplementationConfig`
- Registration: `veomni/kernels/_kernels/moe_experts/__init__.py`
- Model integration: `veomni/models_kernel/transformers/*/*_patch_gen_config.py`
- Plumbing: `veomni/models_kernel/auto.py` — `build_foundation_model(kernels_implementation=...)`

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

## 7. Comparison with Transformers v5 Kernel Selection

VeOmni targets Transformers 5.9.0, whose kernel selection APIs replace the
ad-hoc patching used in earlier versions. This section compares VeOmni's
approach (Sections 1-6 above) with the four
mechanisms available in Transformers v5, using `Qwen3MoE` and `Qwen3.5MoE` as
reference models.

### 7.1 Transformers v5 Mechanisms Overview

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

### 7.2 Side-by-Side Comparison

#### RMSNorm

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | An instance-local `VeomniKernel` selects `liger_kernel`, `npu`, or a model-specific `triton` row | `@use_kernel_forward_from_hub("RMSNorm")` decorator on `Qwen3MoeRMSNorm`; at `model.kernelize()` time the `kernels` library downloads and swaps in `LigerRMSNorm` from `kernels-community/liger_kernels` |
| **Config** | `OpsImplementationConfig.rms_norm_implementation` field (default `"liger_kernel"` on GPU) | `USE_HUB_KERNELS` env var + `model.kernelize()` call |
| **When** | Model registration (import time) | Deferred — `kernelize()` after model init |
| **SP support** | N/A (norm is local) | N/A |
| **Qwen3.5 MoE gap** | Covered by the `rms_norm/qwen3_5` registry variant, including offset-aware Liger and NPU kernels | **Not annotated.** `Qwen3_5MoeRMSNorm` uses `weight * (1.0 + self.weight)` (offset-by-1 convention, weight init to zeros) instead of the standard `self.weight * x` (weight init to ones). No `@use_kernel_forward_from_hub("RMSNorm")` decorator. Standard `LigerRMSNorm` cannot replace it without accounting for the `+1.0` offset. |

#### Rotary Position Embedding (RoPE)

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | An instance-local `VeomniKernel` selects `liger_kernel`, `npu`, or a model-specific `triton` row | `@use_kernel_func_from_hub("rotary_pos_emb")` on the `apply_rotary_pos_emb` function; `kernels` library downloads `apply_rotary_transformers` from `kernels-community/rotary`. The function is also attached to the Attention module via `@use_kernelized_func(apply_rotary_pos_emb)` so `kernelize()` can find it. |
| **Config** | `OpsImplementationConfig.rotary_pos_emb_implementation` field (default `"liger_kernel"` on GPU) | `USE_HUB_KERNELS` env var |
| **When** | Model registration (import time) | Import time (decorator) + `kernelize()` |
| **Qwen3.5 MoE gap** | Covered on NPU by the `rope/partial` registry variant | **Partially annotated.** `apply_rotary_pos_emb` in `Qwen3_5MoeAttention` is annotated with `@use_kernelized_func` but **not** with `@use_kernel_func_from_hub("rotary_pos_emb")`. This is because Qwen3.5 MoE uses *partial RoPE* (`partial_rotary_factor < 1.0`): it splits Q/K into rotary and pass-through parts, applies RoPE only to the rotary part, then concatenates. The standard hub kernel `apply_rotary_transformers` does not handle this split-and-concat pattern. A dedicated partial-RoPE kernel could still be used. |

#### Attention

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | `apply_veomni_attention_patch()` registers SP-aware variants (`veomni_flash_attention_2`, etc.) into `ALL_ATTENTION_FUNCTIONS` | Same `ALL_ATTENTION_FUNCTIONS` registry. Additionally supports hub-based attention kernels via `attn_implementation="kernels-community/flash-mla"` syntax (loaded by `load_and_register_attn_kernel()`). |
| **Config** | `OpsImplementationConfig.attn_implementation` | `config._attn_implementation` (set via `AutoModel.from_pretrained(attn_implementation=...)`) |
| **SP rewrite** | `__post_init__` rewrites `flash_attention_2` → `veomni_flash_attention_2` | No SP support — upstream Transformers does not handle Ulysses SP |
| **Compatibility** | VeOmni registers into the **same** `ALL_ATTENTION_FUNCTIONS` registry that Transformers uses, so the two are compatible by design |

#### MoE Experts

| | VeOmni | Transformers v5 |
|---|--------|----------------|
| **Mechanism** | Each patched experts module constructs `VeomniKernel("moe_experts", variant, impl)` in `__init__` and always calls that handle in `forward`; the selected row may be eager, Triton, Quack, NPU, or MLU. | `@use_experts_implementation` decorator on `Qwen3MoeExperts` class; at forward time dispatches via `ALL_EXPERTS_FUNCTIONS.get_interface(config._experts_implementation, original_forward)`. Built-in implementations: `"batched_mm"` (BMM-based), `"grouped_mm"` (PyTorch `torch.nn.functional.grouped_mm`, requires PT 2.9+). |
| **Config** | `OpsImplementationConfig.moe_implementation` (`"eager"` / `"triton"` / `"quack"` / `"npu"` / `"mlu"`) | `config._experts_implementation` (`"eager"` / `"batched_mm"` / `"grouped_mm"`) |
| **EP support** | `triton` and `npu` paths support Expert Parallelism via VeOmni's EP sharding | `batched_mm` handles invalid expert IDs (sentinel `>= num_experts`) for EP compatibility |
| **When** | Handle resolution in model `__init__`; dispatch at forward time | Decorator at class definition time; dispatch at forward time |

**Note:** Transformers v5 hardcodes two MoE experts implementations (`batched_mm` and `grouped_mm`) and does not expose a registration interface for external fused kernels, so VeOmni models call their local `VeomniKernel` handles rather than routing through `ALL_EXPERTS_FUNCTIONS`.

### 7.3 Gaps — What Transformers v5 Does NOT Cover

The following areas have kernel selection in VeOmni but **no corresponding
mechanism** in Transformers v5:

#### 1. Fused Cross-Entropy Loss

Transformers v5 uses a `loss_function` property on `PreTrainedModel` that looks
up `LOSS_MAPPING[self.loss_type]` — this returns a standard PyTorch
`F.cross_entropy`-based loss. There is no decorator, no hub kernel, and no
env-var-based kernel swap for the loss function.

VeOmni's `models_kernel` stack leaves that global mapping untouched. Generated
model classes construct a local `VeomniKernel` and bind it to
`models_kernel.loss_utils.ForCausalLMLoss` (or the sequence-classification
helper). The fused Liger and chunked implementations compute loss without
materializing the full logits tensor, while wrapper-only input policy remains
outside the raw registry.

#### 2. MoE Load-Balancing Auxiliary Loss

Both Qwen3MoE and Qwen3.5MoE in Transformers v5 include a standalone
`load_balancing_loss_func()` that computes the Switch Transformer auxiliary
loss. This function is called directly in `Qwen3MoeForCausalLM.forward()` —
there is no kernel selection, no registry, and no hub kernel for it.

VeOmni adds a configurable Triton implementation through an instance-local
`VeomniKernel`. `models_kernel/loss_utils/load_balancing_loss.py` preserves the
HF input policy while the registered eager/Triton kernels operate only on a
concatenated `[N, E]` tensor. Transformers itself still has no corresponding
selection surface for this function.

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



### 7.4 Summary Table

| Component | VeOmni mechanism | Transformers v5 mechanism | Compatible? | Gap |
|-----------|-----------------|--------------------------|:-----------:|-----|
| RMSNorm | Instance-local `VeomniKernel` with variants | `@use_kernel_forward_from_hub` | Parallel — both can apply | VeOmni covers Qwen3.5's `+1` variant explicitly |
| RoPE | Instance-local `VeomniKernel` with variants | `@use_kernel_func_from_hub` + `@use_kernelized_func` | Parallel | VeOmni adds an NPU partial-RoPE variant |
| SwiGLU MLP | Instance-local `VeomniKernel` | Not annotated in MoE models (MLP is per-expert, not standalone) | VeOmni only | — |
| Attention | `ALL_ATTENTION_FUNCTIONS` (shared registry) | `ALL_ATTENTION_FUNCTIONS` (same registry) | Yes | VeOmni adds SP wrapping |
| MoE experts | `apply_veomni_fused_moe_patch` (Triton/Quack) | `@use_experts_implementation` (batched_mm/grouped_mm) | No — different dispatch paths | VeOmni uses custom Triton kernels; HF uses PyTorch native `grouped_mm` |
| Cross-entropy | Instance-local `VeomniKernel` + model helper | `LOSS_MAPPING` (standard `F.cross_entropy`) | VeOmni only | HF has no fused loss selection |
| MoE aux loss | Instance-local eager/Triton `VeomniKernel` + model helper | Eager `load_balancing_loss_func` | VeOmni only | HF has no fused selection surface |
| RMSNormGated | Instance-local `VeomniKernel` (`fla`/`npu`/`eager`) | Hard-coded `fla.modules.FusedRMSNormGated` if `fla` is installed, else eager | Different dispatch | VeOmni adds explicit hardware selection |

---

## Full Config Example

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: triton
    cross_entropy_loss_implementation: liger_kernel
    rms_norm_implementation: liger_kernel
    swiglu_mlp_implementation: eager           # disable Liger for MLP only
    rotary_pos_emb_implementation: liger_kernel
    load_balancing_loss_implementation: triton
```
