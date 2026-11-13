# Float8 training (tensorwise and rowwise, LLM and DiT)

VeOmni can opt selected `torch.nn.Linear` modules into torchao's dynamic float8
forward and backward GEMMs (`tensorwise`, `rowwise`, or `rowwise_with_gw_hp`)
across both dense LLM (`qwen3`, `llama`) and DiT (`WanTransformer3DModel`,
`QwenImageTransformer2DModel`, `LTXVideoTransformerModel`, `MiniMaxH3DiTModel`)
backbones. Parameters and optimizer updates remain FP32; FSDP2 unshards
parameters to BF16 and reduces gradients in FP32. The feature is disabled by
default and does not import torchao when disabled.

This implements the float8 compute slices of [RFC #1360](https://github.com/ByteDance-Seed/VeOmni/issues/1360)
and [Tracking Issue #1366](https://github.com/ByteDance-Seed/VeOmni/issues/1366).

## Install and configure

The torchao backend is an optional extra, not part of the `gpu` extra, so GPU
jobs that never enable float8 do not install it. Add the `float8` extra to the
GPU environment:

```bash
uv sync --extra gpu --extra float8 --dev
```

Enabling float8 without the extra raises `ImportError: float8 training requires
torchao==0.18.0, which is an optional extra: uv sync --extra gpu --extra float8 --dev`
before any module is converted. The exact pin is checked because torchao's
`convert_to_float8_training` internals are not stable across minor versions.

For dense Qwen3 or Llama with rowwise float8:

```yaml
model:
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
        float8:
          enable: true
          recipe: rowwise
          target_modules:
            - model.layers.*.mlp.gate_proj
            - model.layers.*.mlp.up_proj
            - model.layers.*.mlp.down_proj
```

For `WanTransformer3DModel` DiT blocks (select only `nn.Linear` leaves; do not
match the `to_out` `ModuleList` container or its dropout child):

```yaml
model:
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
        float8:
          enable: true
          recipe: rowwise
          target_modules:
            - blocks.*.attn1.to_q
            - blocks.*.attn1.to_k
            - blocks.*.attn1.to_v
            - blocks.*.attn1.to_out.0
            - blocks.*.attn2.to_q
            - blocks.*.attn2.to_k
            - blocks.*.attn2.to_v
            - blocks.*.attn2.to_out.0
            - blocks.*.ffn.net.0.proj
            - blocks.*.ffn.net.2
```

Model-specific eligible DiT projection patterns:
- `WanTransformer3DModel`: `blocks.*.attn1.{to_q,to_k,to_v,to_out.0}`, `blocks.*.attn2.{to_q,to_k,to_v,to_out.0}`, `blocks.*.attn2.{add_k_proj,add_v_proj}` (only present when `added_kv_proj_dim` is set, i.e. I2V image cross-attention), `blocks.*.ffn.net.{0.proj,2}`
- `QwenImageTransformer2DModel`: `transformer_blocks.*.attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}`, `transformer_blocks.*.{img_mlp,txt_mlp}.net.{0.proj,2}`
- `LTXVideoTransformerModel`: `transformer_blocks.*.{attn1,attn2,audio_attn1,audio_attn2,audio_to_video_attn,video_to_audio_attn}.{to_q,to_k,to_v,to_out.0}`, `transformer_blocks.*.{ff,audio_ff}.net.{0.proj,2}`
- `MiniMaxH3DiTModel`: `dit.blocks.*.attn.{qkv_proj,out_proj}`, `dit.blocks.*.mlp.{fc1,fc2}`, `dit.token_refiner.blocks.*.attn.{qkv_proj,out_proj}`, `dit.token_refiner.blocks.*.mlp.{fc1,fc2}`

## Supported surface

| Property | Support |
|---|---|
| Models | Dense `qwen3`, `llama`; DiT `WanTransformer3DModel`, `QwenImageTransformer2DModel`, `LTXVideoTransformerModel`, `MiniMaxH3DiTModel` |
| Recipes | `tensorwise` (default), `rowwise` (token-wise activation × channel-wise weight scaling), `rowwise_with_gw_hp` (rowwise forward, token-wise `grad_output` × tensorwise weight for `dX`, and uncast high-precision operands for `dW`) |
| Modules | Explicitly selected plain `nn.Linear` transformer block projections with `in_features` and `out_features` divisible by 16 |
| Excluded | LLM `lm_head`; DiT patchify, timestep/caption/condition embedders, AdaLN modulation layers, and final output projections |
| Hardware | NVIDIA CUDA SM89 or newer (Ada, Hopper, Blackwell) |
| Parameters / optimizer | FP32 master parameters and standard optimizer updates |
| Checkpoints | Existing HF weight loading, DCP model/optimizer resume, and floating-point HF export |

Configuration is defined by [Float8TrainingConfig](../../veomni/arguments/arguments_types.py#L604),
validated by [validate_float8_training](../../veomni/distributed/float8.py#L51),
and applied by [convert_float8_linears](../../veomni/distributed/float8.py#L86)
inside [parallelize_model_fsdp2](../../veomni/distributed/torch_parallelize.py#L429)
before FSDP2 wraps transformer blocks.

### Interactions

- **Fused SwiGLU MLP**: converting a model's MLP `gate_proj` / `up_proj` /
  `down_proj` replaces them with torchao's `Float8Linear`, an `nn.Linear`
  subclass whose `forward` runs the float8 GEMMs. The fused `swiglu_mlp` row
  (including the GPU default `liger_kernel`) reads projection weights directly,
  so for those MLPs it is bypassed and the model runs the unfused
  `down_proj(act_fn(gate_proj(x)) * up_proj(x))`. A quantizing
  `swiglu_mlp_implementation` (`qat`) on converted projections is refused.
- **PEFT interop**: `peft==0.18.1` (from the `gpu` extra) imports
  `LinearActivationQuantizedTensor` from torchao when torchao is installed, and
  torchao 0.18.0 no longer provides it. With the `float8` extra installed,
  PEFT-format LoRA interop (for example
  `tests/lora/test_veomni_lora_native.py::test_peft_bidirectional_interop`)
  fails with `ImportError`. Native `veomni.lora` does not import PEFT, and
  float8 training rejects LoRA anyway, so float8 runs are unaffected; use an
  environment without the extra for PEFT interop.

## Recipe selection

- **`tensorwise`**: Uses per-tensor dynamic scaling across all three GEMMs (`output`, `dX`, `dW`).
- **`rowwise_with_gw_hp`**: Uses token-wise activation scaling and channel-wise weight scaling for the forward GEMM, token-wise `grad_output` scaling with tensorwise weight scaling for the activation-gradient (`dX`) GEMM, and skips FP8 casts for the weight-gradient (`dW`) operands.
- **`rowwise`**: Uses rowwise scaling across all three GEMMs (`output`, `dX`, `dW`).
- **`torch.compile` status**: `model.accelerator.torch_compile.enable=True` is currently rejected by `validate_float8_training()` (`ValueError: float8 training with torch.compile is not yet validated.`), so supported FSDP2 float8 training runs in eager mode (`torch_compile.enable=False`).

## Tests and CI coverage

`tests/models/test_float8_training.py` holds the float8 tests. Configuration
validation, the lazy-import guard, and the missing/mismatched-torchao errors
run on CPU without torchao. The GPU cases (kernel dispatch, numerics, FSDP2
training, DCP resume and HF export) call `pytest.importorskip("torchao.float8")`
and also need an NVIDIA SM89+ device, so they **skip** on any machine where the
`float8` extra is not installed — including CI jobs that install only
`--extra gpu`. A skip there is not a pass: the tests stay in-tree so whoever
installs the extra on SM89+ hardware runs them:

```bash
uv sync --extra gpu --extra float8 --dev
pytest tests/models/test_float8_training.py
```
