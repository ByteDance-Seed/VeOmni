# Float8 training (tensorwise and rowwise)

VeOmni runs torchao's dynamic float8 forward and backward GEMMs (`tensorwise`,
`rowwise`, or `rowwise_with_gw_hp`) directly inside the `linear` and `gated_mlp`
op families via the `torchao` row (`dtype: fp8`). Parameters and optimizer
updates remain FP32; FSDP2 unshards parameters to BF16 and reduces gradients in
FP32; every projection stays a plain `torch.nn.Linear` with no module swapping
and no extra config patches on `mixed_precision` or trainers.

When `torchao` is not selected, `torchao` is never imported.

## Install and configure

The torchao backend is an optional extra, not part of the `gpu` extra, so GPU
jobs that never select `torchao` do not install it. Add the `float8` extra to
the GPU environment:

```bash
uv sync --extra gpu --extra float8 --dev
```

Running `dtype: fp8` without the extra raises `ImportError: float8 training
requires torchao==0.18.0, which is an optional extra: uv sync --extra gpu
--extra float8 --dev`. The exact pin is checked because torchao's float8
training internals are not stable across minor versions.

For dense Qwen3 or Llama with rowwise float8 on both attention and MLP
projections:

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: torchao
      kwargs:
        dtype: fp8
        recipe: rowwise
    gated_mlp_implementation:
      name: torchao
      kwargs:
        dtype: fp8
        recipe: rowwise
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
```

Use per-site `overrides` to enable float8 only on selected layers or
projections (for example, keeping layer 0 or `k_proj`/`v_proj` in BF16):

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: eager
      overrides:
        - site: 'layers\.(?:[1-9]\d*)\.(q_proj|o_proj)'
          name: torchao
          kwargs:
            dtype: fp8
            recipe: rowwise
    gated_mlp_implementation:
      name: liger_kernel
      overrides:
        - site: 'layers\.(?:[1-9]\d*)\.mlp'
          name: torchao
          kwargs:
            dtype: fp8
            recipe: rowwise
```

## Supported surface

| Property | Support |
|---|---|
| Op families | `linear` (`standard` variant: Qwen3, Llama, DeepSeek-V4 attention/indexer projections) and `gated_mlp` (`standard` variant: all models binding `VeomniOp("gated_mlp", "standard", ...)`) |
| Formats on `torchao` | `dtype: fp8` (default `emulated: false`, native `aten::_scaled_mm`), `dtype: nvfp4`, `dtype: mxfp4` (default `emulated: true`, QAT fake quantization) |
| Float8 recipes | `tensorwise` (default), `rowwise` (token-wise activation × channel-wise weight scaling), `rowwise_with_gw_hp` (rowwise forward, token-wise `grad_output` × tensorwise weight for `dX`, high-precision `dW`) |
| Shapes | Plain `nn.Linear` projections with `in_features` and `out_features` divisible by 16 |
| Excluded | `lm_head` (outside `linear` / `gated_mlp` rows) |
| Hardware | Native (`emulated: false`): NVIDIA CUDA SM89 or newer (Ada, Hopper, Blackwell). Emulated (`emulated: true`): CPU or CUDA SM80+ |
| Parameters / optimizer | Plain FP32 master `nn.Linear` parameters and standard optimizer updates |
| Checkpoints | Existing HF weight loading, DCP model/optimizer resume, and floating-point HF export |

The functional primitive lives in
[`veomni/ops/qat/torchao_fp8.py`](../../veomni/ops/qat/torchao_fp8.py) and is
invoked by [`veomni/ops/kernels/linear/standard/torchao.py`](../../veomni/ops/kernels/linear/standard/torchao.py)
and [`veomni/ops/kernels/gated_mlp/standard/torchao.py`](../../veomni/ops/kernels/gated_mlp/standard/torchao.py).

### Interactions

- **Wrapped projections (LoRA / `basic_modules`)**: `torchao` reads the plain
  projection's `weight` and `bias` directly. Wrapped projections (`LoraLinear`)
  or `nn.Linear` subclasses (`FSDPLinear` from `basic_modules`) are refused by
  `linear._module.resolve_projection` and `gated_mlp._module.render` rather
  than silently bypassing them. Use `eager` (or per-site `overrides`) for
  adapted layers.
- **PEFT interop**: `peft==0.18.1` (from the `gpu` extra) imports
  `LinearActivationQuantizedTensor` from torchao when torchao is installed, and
  torchao 0.18.0 no longer provides it. Native `veomni.lora` does not import
  PEFT; use an environment without the `float8` extra for PEFT interop tests.

## Recipe selection

- **`tensorwise`**: Per-tensor dynamic scaling across all three GEMMs (`output`, `dX`, `dW`).
- **`rowwise_with_gw_hp`**: Token-wise activation scaling and channel-wise weight scaling for forward, token-wise `grad_output` scaling with tensorwise weight scaling for `dX`, and high-precision operands for `dW`.
- **`rowwise`**: Rowwise scaling across all three GEMMs (`output`, `dX`, `dW`).

## Tests and CI coverage

`tests/models/test_float8_training.py` holds the float8 tests. Config
round-tripping, per-site handle resolution, lazy-import guards, shape/recipe
validation, hardware checks, and missing/mismatched-torchao errors run on CPU
without torchao. The GPU cases (`aten::_scaled_mm` dispatch, FSDP2 training,
DCP resume, and HF export) call `pytest.importorskip("torchao.float8")` and
require an NVIDIA SM89+ device:

```bash
uv sync --extra gpu --extra float8 --dev
pytest tests/models/test_float8_training.py
```
