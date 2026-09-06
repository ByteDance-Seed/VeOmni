# `veomni.ops` — Legacy Model-Integration Dispatch

This package retains the model-integration hooks that have not yet migrated to
`veomni.kernels`. Tensor-native kernel implementations and their registry live
under `veomni/kernels`; model-facing loss policy lives under
`veomni/models_kernel/loss_utils`.

## Directory layout

```
veomni/ops/
├── config/                 Dispatch infrastructure (no kernels here)
│   ├── registry.py         Legacy OpSpec / BackendSpec / OpScope dispatch
│   └── singleton.py        get_ops_config / set_ops_config — bridges the
│                           resolved config from BaseTrainer to device_patch.py
├── kernels/                Remaining legacy model-integration implementations
│   ├── deepseek_sparse_attention/
│   └── deepseek_v4/        Legacy model-specific helpers
├── platform/               Platform-specific runtime patches
│   └── npu/                HCCL pre-mul sum patch
└── batch_invariant_ops/    Opt-in deterministic-mode toggle
```

## Dispatch model

Legacy selection is driven by `OpsImplementationConfig` fields
(`model.ops_implementation.*` in YAML). The remaining dispatch scopes differ
by when and where the integration is bound:

| Scope | Who binds | When | What gets replaced |
|-------|-----------|------|--------------------|
| **import-time** | `apply_ops_patch()` | `import veomni` | Registers VeOmni attention kernels in HF's `ALL_ATTENTION_FUNCTIONS`. Gated by `MODELING_BACKEND`. |
| **PER_MODEL** | `apply_per_model_patches()` in each model's `device_patch.py` | During `build_foundation_model()` | `setattr(hf_module, "<ClassOrFuncName>", …)` on the HF modeling module (different class name per model). |
| **build-time** | `apply_veomni_fused_moe_patch()` | During `build_foundation_model()` | `veomni.ops.kernels.moe._fused_moe_forward`; NPU auto-overrides to the NPU group-gemm kernel. |

### All kernels at a glance

| Kernel | Config key | Scope | Default | Available backends |
|---|---|:-:|---|---|
| Attention | `attn_implementation` | import-time | `flash_attention_2` | `eager`, `sdpa`, `flash_attention_2/3/4`, `flex_attention`, `magi_attention`, `native-sparse` |
| RMSNorm | `rms_norm_implementation` | PER_MODEL | `eager` | `liger_kernel`, `npu`, `triton`\* |
| Rotary pos emb | `rotary_pos_emb_implementation` | PER_MODEL | `eager` | `liger_kernel`, `npu`, `triton`\* |
| SwiGLU MLP | `swiglu_mlp_implementation` | PER_MODEL | `eager` | `liger_kernel` |
| Fused MoE | `moe_implementation` | build-time | `eager` | `eager`, `triton` (group-gemm, SM70+ GPU or MLU), `quack` (CUTLASS/CuTe, SM90+), `npu` (Ascend), `mlu` (Apex grouped-GEMM). Mismatches raise instead of falling back. |

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
| `flex_attention` | PyTorch FlexAttention | Native `BlockMask`; compiled CUDA execution for training |
| `magi_attention` | `magi-attention==1.1.1`, NVIDIA SM90+ | Native `MagiAttentionMask`; CP1 FFA with optional Ulysses through the SM90 CUTLASS overlay or SM100+ CUTE DSL/JIT backend. |
| `moe_implementation=triton` | Triton, SM70+ GPU or MLU | `is_fused_moe_available()` |
| `moe_implementation=quack` | `quack` package, SM90+ | `is_quack_gemm_available()` |
| `moe_implementation=npu` | `torch_npu` + Ascend NPU | `is_torch_npu_available()` |
| `moe_implementation=mlu` | `apex` grouped-GEMM on MLU | `is_apex_mlu_available()` |
| `mhc_implementation=tilelang` | `tile-kernels==1.0.0`, BF16, NVIDIA SM90+ | `KernelSpec(HardwareRequirement(..., min_compute_capability=90))` |

#### Installing MagiAttention

The GPU extra installs MagiAttention and its SM100+ CUTE DSL/JIT backend:

```bash
uv sync --extra gpu --dev
```

SM90 additionally requires the precompiled CUTLASS overlay:

```bash
bash scripts/kernel/install_magi_sm90.sh
```

The verified default enables BF16/FP16 inputs, the hdim128 bucket, and nfunc 1/3/5. Run the installer with `--help` to see optional dtype, head-dimension, nfunc, and compiler-concurrency overrides. SM80 and older GPUs are unsupported.

### Per-model PER_MODEL coverage

Each model's `device_patch.py` or patchgen config binds the PER_MODEL ops to
the HF modeling symbols it uses:

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
| `deepseek_v4` | `DeepseekV4RMSNorm` + `DeepseekV4UnweightedRMSNorm` | *(eager only)* | `DeepseekV4MLP` shared experts | Patchgen OpSlots; routed experts remain on clamp-aware fused MoE |
| `wan` (DiT) | `RMSNorm` | `rope_apply` | *(n/a)* | `triton` RMSNorm/rotary via `extra_backends`; attention block wired via `custom_patches` |

### Full YAML example

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused
    rms_norm_implementation: liger_kernel
    rotary_pos_emb_implementation: liger_kernel
    swiglu_mlp_implementation: eager   # keep HF MLP even when Liger is on
```

See `docs/design/kernel_selection.md` for the user-facing lifecycle diagram.
See `docs/transformers_v5/veomni_fused_attention.md` for the Flash/Flex
facade, native BlockMask contract, and Ulysses behavior.

### DeepSeek V4 library kernels

`kernels/deepseek_v4/` contains model-specific library ops adapted from the
Apache-2.0 `radixark/miles` implementation: TileLang sparse-attention and
Lightning Indexer forward/backward kernels, block-wise FP8 activation
quantization, and a BF16-input/FP32-accumulation linear autograd function.
`veomni/kernels/_kernels/mhc/` adapts TileKernels' training-capable DeepSeek
V4 mHC pre, post, and head kernels behind instance-local `VeomniKernel`
handles.
The package does not import TileLang eagerly, so CPU and NPU installations can
still import VeOmni. Callers that use a TileLang entry point must have the GPU
extra installed. The GPU extra pins `tilelang==0.1.9` and
`tile-kernels==1.0.0`; FlashQLA 0.1.2 pins the same TileLang version, so all
three TileLang consumers share one validated build.

DeepSeek-V4 selects these kernels with `dsa_indexer_implementation: tilelang` and
`dsa_attention_implementation: tilelang`. Both default to `eager`; unsupported
cache, position, or dropout layouts retain the upstream eager implementation.
DeepSeek V4 selects the mHC adapters with `mhc_implementation: tilelang`; once
selected, unsupported dtype, layout, or hardware raises instead of falling
back to eager.

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

### Load-balancing loss ownership

Load-balancing loss no longer has an `ops` facade or process-global function
pointer. Its eager and Triton implementations are registered as tensor-native
`[N, E]` kernels under `veomni/kernels/_kernels/loss/load_balancing_loss`.
Each MoE model constructs an instance-local `VeomniKernel` and binds it to the
HF-shaped helper in `veomni/models_kernel/loss_utils/load_balancing_loss.py`.
That helper handles `None`, per-layer tuple concatenation, and the optional
attention mask; the raw kernel only performs loss math.

Cross-entropy follows the same ownership split: token-level eager, chunked,
and Liger implementations are registered under
`veomni/kernels/_kernels/loss/cross_entropy_loss`; causal shifting, sequence
classification policy, SP reduction, log-probs, and distillation routing live
in `veomni/models_kernel/loss_utils`. Models bind a local `VeomniKernel`; this
package no longer owns a facade or mutates Transformers' `LOSS_MAPPING`.

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
