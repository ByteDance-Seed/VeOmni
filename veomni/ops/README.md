# `veomni.ops` — Kernel Registry and Dispatch

This package houses every optimized kernel VeOmni selects at runtime
(attention, cross-entropy loss, RMSNorm, RoPE, SwiGLU MLP, fused MoE, …) and
the dispatch machinery that picks the right implementation based on
`OpsImplementationConfig`.

## Directory layout

```
veomni/ops/
├── config/                 Dispatch infrastructure (no kernels here)
│   ├── registry.py         OpSpec / BackendSpec / OpScope + register_op,
│   │                       apply_global_ops, apply_per_model_patches
│   └── singleton.py        get_ops_config / set_ops_config — bridges the
│                           resolved config from BaseTrainer to device_patch.py
├── kernels/                Kernel implementations, one subpackage per op
│   ├── attention/          Flash attention v2/3/4 + SP-aware wrappers
│   ├── cross_entropy/      eager / liger / npu-chunk loss (+ ForCausalLMLoss)
│   ├── load_balancing_loss/  eager + triton fused kernel
│   ├── rms_norm/           Liger / NPU / triton batch-invariant
│   ├── rotary/             Liger / NPU / deterministic / Wan Triton
│   ├── swiglu/             Liger SwiGLU MLP
│   └── moe/                Fused MoE + _kernels/ (group_gemm, quack_gemm)
├── platform/               Platform-specific runtime patches
│   └── npu/                HCCL pre-mul sum patch
└── batch_invariant_ops/    Opt-in deterministic-mode toggle
```

## Dispatch model

All kernel selection is driven by `OpsImplementationConfig` fields
(`model.ops_implementation.*` in YAML). There are **four** dispatch scopes
depending on when and where the kernel is bound:

| Scope | Who binds | When | What gets replaced |
|-------|-----------|------|--------------------|
| **import-time** | `apply_ops_patch()` | `import veomni` | Registers VeOmni attention kernels in HF's `ALL_ATTENTION_FUNCTIONS`. Gated by `MODELING_BACKEND`. |
| **LOSS_MAPPING** | `install_loss_mapping()` via `apply_ops_config()` | Before model build, in `BaseTrainer` | `LOSS_MAPPING["ForCausalLM"/"ForConditionalGeneration"/"ForSequenceClassification"]` bound to `partial(<wrapper>, cross_entropy_fn=<impl>)`. |
| **GLOBAL** | `apply_global_ops()` via `apply_ops_config()` | Before model build, in `BaseTrainer` | Module-level function pointer shared by all models (e.g. `veomni.ops.kernels.load_balancing_loss._load_balancing_loss`). |
| **PER_MODEL** | `apply_per_model_patches()` in each model's `device_patch.py` | During `build_foundation_model()` | `setattr(hf_module, "<ClassOrFuncName>", …)` on the HF modeling module (different class name per model). |
| **build-time** | `apply_veomni_fused_moe_patch()` | During `build_foundation_model()` | `veomni.ops.kernels.moe._fused_moe_forward`; NPU auto-overrides to the NPU group-gemm kernel. |

### All kernels at a glance

| Kernel | Config key | Scope | Default | Available backends |
|---|---|:-:|---|---|
| Attention | `attn_implementation` | import-time | `flash_attention_2` | `eager`, `sdpa`, `flash_attention_2/3/4`, `native-sparse` |
| Cross-entropy loss | `cross_entropy_loss_implementation` | LOSS_MAPPING | `eager` | `eager`, `liger_kernel`, `npu` (chunked loss) |
| Load-balancing loss | `load_balancing_loss_implementation` | GLOBAL | `eager` | `eager`, `triton` |
| RMSNorm | `rms_norm_implementation` | PER_MODEL | `eager` | `liger_kernel`, `npu`, `triton`\* |
| Rotary pos emb | `rotary_pos_emb_implementation` | PER_MODEL | `eager` | `liger_kernel`, `npu`, `triton`\* |
| SwiGLU MLP | `swiglu_mlp_implementation` | PER_MODEL | `eager` | `liger_kernel` |
| Fused MoE | `moe_implementation` | build-time | `eager` | `eager`, `fused_triton` (group-gemm, SM70+), `fused_quack` (CUTLASS/CuTe, SM90+), `fused_npu` (Ascend). Mismatches raise instead of falling back. |

\* The `triton` backend is registered per-model via `extra_backends`: DeepSeek
V3 exposes a batch-invariant RMSNorm + deterministic RoPE, and Wan exposes its
own Triton RMSNorm/rotary. See the per-model table below.

### Backend availability requirements

| Backend | Requirement | How it's checked |
|---|---|---|
| `eager` | — | Always available |
| `liger_kernel` | `liger-kernel` package | `BackendSpec.requires=("liger_kernel",)` → `is_liger_kernel_available()` |
| `npu` | `torch_npu` + Ascend NPU | `BackendSpec.requires=("torch_npu",)` → `is_torch_npu_available()` |
| `triton` | Triton + CUDA | Validated by the model `extra_backends` registration |
| `flash_attention_2/3/4` | `flash-attn` / `flash-attn-interface` / `flash-attn.cute` | Validated in `OpsImplementationConfig.__post_init__` |
| `moe_implementation=fused_triton` | Triton, SM70+ | `is_fused_moe_available()` |
| `moe_implementation=fused_quack` | `quack` package, SM90+ | `is_quack_gemm_available()` |
| `moe_implementation=fused_npu` | `torch_npu` + Ascend NPU | `is_torch_npu_available()` |

### Per-model PER_MODEL coverage

Each model's `device_patch.py` binds the three PER_MODEL ops to the HF
modeling symbols it uses:

| Model | `rms_norm` target | `rotary_pos_emb` target | `swiglu_mlp` target | Extras |
|---|---|---|---|---|
| `llama` | `LlamaRMSNorm` | `apply_rotary_pos_emb` | `LlamaMLP` | — |
| `qwen2` | `Qwen2RMSNorm` | `apply_rotary_pos_emb` | `Qwen2MLP` | — |
| `qwen3` | `Qwen3RMSNorm` | `apply_rotary_pos_emb` | `Qwen3MLP` | — |
| `qwen3_moe` | `Qwen3MoeRMSNorm` | `apply_rotary_pos_emb` | `Qwen3MoeMLP` | — |
| `seed_oss` | `SeedOssRMSNorm` | `apply_rotary_pos_emb` | `SeedOssMLP` | — |
| `qwen2_vl` | `Qwen2RMSNorm` | `apply_multimodal_rotary_pos_emb` | `Qwen2MLP` | `rotary_pos_emb.npu` disabled; vision RoPE via `custom_patches` |
| `qwen3_vl` | `Qwen3VLTextRMSNorm` | `apply_rotary_pos_emb` | *(n/a)* | `liger_kernel` disabled for RMSNorm/RoPE; vision RoPE via `custom_patches` |
| `deepseek_v3` | `DeepseekV3RMSNorm` | `apply_rotary_pos_emb` | `DeepseekV3MLP` | `triton` adds batch-invariant RMSNorm + deterministic RoPE (patches `DeepseekV3RotaryEmbedding.forward` via `target_override`) |
| `wan` (DiT) | `RMSNorm` | `rope_apply` | *(n/a)* | `triton` RMSNorm/rotary via `extra_backends`; attention block wired via `custom_patches` |

### Full YAML example

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused
    cross_entropy_loss_implementation: liger_kernel
    load_balancing_loss_implementation: triton
    rms_norm_implementation: liger_kernel
    rotary_pos_emb_implementation: liger_kernel
    swiglu_mlp_implementation: eager   # keep HF MLP even when Liger is on
```

See `docs/design/kernel_selection.md` for the user-facing lifecycle diagram.

---

## Recipe 1: Add a new backend to an existing op

Example: add a `triton` backend for `rms_norm` on GPU.

1. Put the kernel under `veomni/ops/kernels/rms_norm/triton.py`, exporting a
   callable (module class or function, whichever matches the op's shape).
2. Register it by extending the `OpSpec.backends` dict in
   `veomni/ops/kernels/rms_norm/__init__.py`:

   ```python
   "triton": BackendSpec(
       entry="veomni.ops.kernels.rms_norm.triton:TritonRMSNorm",
   ),
   ```

3. (Optional) For a *model-specific* override (doesn't belong in the global
   registry), pass it via `extra_backends` from that model's
   `device_patch.py`:

   ```python
   apply_per_model_patches(
       hf_module=hf_deepseek_v3,
       model_name="DeepseekV3",
       targets={"rms_norm": "DeepseekV3RMSNorm"},
       extra_backends={
           "rms_norm": {
               "triton": BackendSpec(
                   entry="veomni.ops.kernels.rms_norm.triton_batch_invariant:BatchInvariantRMSNorm",
                   replace_forward=True,
               ),
           },
       },
   )
   ```

4. Users pick it with `model.ops_implementation.rms_norm_implementation=triton`.
   The registry validates availability via `BackendSpec.requires`.

---

## Recipe 2: Add a brand-new op

Example: add `layer_norm` as a per-model op.

1. Add a field to `OpsImplementationConfig`
   (`veomni/arguments/arguments_types.py`):

   ```python
   layer_norm_implementation: str = "eager"
   ```

2. Create `veomni/ops/kernels/layer_norm/` with:
   - `<backend>.py` files containing the actual kernels (e.g.
     `liger.py`, `triton.py`).
   - `__init__.py` that calls `register_op`:

     ```python
     from ...config.registry import BackendSpec, OpScope, OpSpec, register_op

     register_op(
         OpSpec(
             name="layer_norm",
             config_field="layer_norm_implementation",
             label="LayerNorm",
             scope=OpScope.PER_MODEL,
             default="eager",
             backends={
                 "liger_kernel": BackendSpec(
                     entry="veomni.ops.kernels.layer_norm.liger:LigerLayerNorm",
                     requires=("liger_kernel",),
                 ),
             },
         )
     )
     ```

3. Import the subpackage from `veomni/ops/kernels/__init__.py` so registration
   runs on `import veomni`.

4. Reference the op in each model's `device_patch.py`:

   ```python
   apply_per_model_patches(
       hf_module=hf_llama,
       model_name="Llama",
       targets={"layer_norm": "LlamaLayerNorm"},
   )
   ```

### GLOBAL instead of PER_MODEL

For ops that are a single function pointer shared across all models (like
`load_balancing_loss`), set `scope=OpScope.GLOBAL` and provide a
`global_slot="<module>:<attr>"`. `apply_global_ops()` writes the selected
backend to that slot; callers `from ... import <attr>` and call it. See
`kernels/load_balancing_loss/__init__.py` for the full pattern.

Cross-entropy is handled separately via `LOSS_MAPPING` scope (see
`install_loss_mapping` in `kernels/cross_entropy/__init__.py`) — it needs
three distinct wrapper shapes (`ForCausalLM`, `ForConditionalGeneration`,
`ForSequenceClassification`) rather than a single function pointer, so the
GLOBAL slot pattern does not fit.

---

## Edge cases

- **Hardware requirements**: list the import guard in `BackendSpec.requires`
  (`"liger_kernel"` and `"torch_npu"` are supported today; extend
  `_check_requires` in `registry.py` to add more).
- **Replace `.forward` instead of the class** (NPU RMSNorm): set
  `replace_forward=True`.
- **Factory backends** (DeepSeek V3 deterministic RoPE): set
  `entry_is_factory=True`; `entry` becomes a zero-arg callable returning the
  actual replacement.
- **Different target attribute per backend** (DeepSeek V3 Triton RoPE patches
  `DeepseekV3RotaryEmbedding` while Liger/NPU patch `apply_rotary_pos_emb`):
  set `target_override` on the `BackendSpec`.
- **Disable a default backend for one model** (Qwen2-VL has no NPU RoPE
  support for multimodal RoPE): pass `extra_backends={"rotary_pos_emb":
  {"npu": None}}` to `apply_per_model_patches`.
- **Truly one-off patches** (Wan model's custom rotary in forward): use the
  `custom_patches=` callback hook of `apply_per_model_patches`.
