# Low-Precision Training (FP8, MXFP8, FP4 QAT, and DeepGEMM MoE)

VeOmni configures low-precision training **per op family** through
`model.ops_implementation`, rather than through a global quantization switch or
module-swapping patch:

| Field | Low-precision rows | Supported `dtype` | Execution | Hardware / Extra |
| --- | --- | --- | --- | --- |
| `linear_implementation` | `torchao` (`standard` only) | `fp8` / `mxfp8` (`emulated: false`), `nvfp4` / `mxfp4` (`emulated: true`) | Dynamic FP8, block-scaled MXFP8, or FP4 fake-quant | `--extra float8` (`torchao==0.18.0`); SM89+ for native FP8, SM100+ for native MXFP8, CPU / SM80+ for emulated |
| `linear_implementation` | `qat` (`standard`, `grouped`) | `fp8` | Blockwise FP8 fake-quant | SM90+, `tilelang` |
| `linear_implementation` | `deepgemm` (`standard`, `grouped`) | `fp8` only | Native FP8 GEMM (forward, dgrad, wgrad) | Hopper SM90 / Blackwell SM100, `deep_gemm` + `triton` |
| `gated_mlp_implementation` | `torchao` | `fp8` / `mxfp8` (`emulated: false`), `nvfp4` / `mxfp4` (`emulated: true`) | Dynamic FP8, block-scaled MXFP8, or FP4 fake-quant | `--extra float8` (`torchao==0.18.0`); SM89+ for native FP8, SM100+ for native MXFP8, CPU / SM80+ for emulated |
| `gated_mlp_implementation` | `qat` | `fp8` | Blockwise FP8 fake-quant | SM90+, `tilelang` |
| `gated_mlp_implementation` | `deepgemm` (`standard`) | `fp8` only | Native FP8 GEMM (forward, dgrad, wgrad); SiLU / `swish` only | Hopper SM90 / Blackwell SM100, `deep_gemm` + `triton` |
| `moe_implementation` | `deepgemm` | `fp8` only (today) | Native FP8 grouped GEMM (forward, dgrad, wgrad) | Hopper SM90 / Blackwell SM100, `deep_gemm` + `triton` |
| `moe_implementation` | `qat` | `fp8`, `fp4` | Blockwise FP8/FP4 fake-quant + Triton grouped GEMM | SM90+, `tilelang` + `triton` |
| `kv_round_implementation` | `triton`, `eager` | — | Stored KV / indexer activation rounding | `triton`: SM89+; `eager`: device-agnostic |

Every projection stays a plain `torch.nn.Linear` (and every expert stack stays
ordinary floating-point tensors): parameters and optimizer updates remain FP32,
FSDP2 unshards parameters to BF16 and reduces gradients in FP32, and checkpoints
remain standard floating-point weights. The projection operators load their
optional torchao kernels when a torchao row executes.

`lm_head` is outside the `linear` and `gated_mlp` op families and is never
quantized by these selections.

See [op selection](../design/op_selection.md) for the full registry design and
[FP8 MoE training](../design/deepgemm_moe.md) for DeepGEMM MoE internals.

---

## Install

The `torchao` backend (used by `linear_implementation: torchao` and
`gated_mlp_implementation: torchao` for FP8, MXFP8, and FP4 QAT) is an optional
extra so jobs that do not select it avoid pulling it in:

```bash
uv sync --locked --extra gpu --extra float8 --dev
source .venv/bin/activate      # train.sh resolves torchrun from PATH
```

Selecting `torchao` without `torchao` installed raises an explicit error at
model construction. The FP8 and MXFP8 implementations also check the installed
torchao version at runtime because they use version-specific APIs.

> **PEFT interop note**: `peft==0.18.1` (from the `gpu` extra) imports
> `LinearActivationQuantizedTensor` from torchao when torchao is installed,
> which torchao 0.18.0 removed. Native `veomni.lora` does not import PEFT; use
> an environment without the `float8` extra when running PEFT interop tests.

---

## TorchAO Dynamic Float8 (`dtype: fp8`)

The `torchao` row runs torchao's dynamic float8 forward and backward GEMMs
(`aten::_scaled_mm`, default `emulated: false`) directly inside `linear` and
`gated_mlp`.

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: torchao
      kwargs: {dtype: fp8, recipe: rowwise}
    gated_mlp_implementation:
      name: torchao
      kwargs: {dtype: fp8, recipe: rowwise}
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
```

### Float8 recipes

- **`tensorwise`** (default): Per-tensor dynamic scaling across all three GEMMs (`output`, `dX`, `dW`).
- **`rowwise_with_gw_hp`**: Token-wise activation scaling and channel-wise weight scaling for forward, token-wise `grad_output` scaling with tensorwise weight scaling for `dX`, and high-precision operands for `dW`.
- **`rowwise`**: Rowwise scaling across all three GEMMs (`output`, `dX`, `dW`).

### Requirements

- **Hardware**: Native (`emulated: false`) requires NVIDIA CUDA SM89+ (Ada, Hopper, Blackwell). Emulated (`emulated: true`) runs on CPU or CUDA SM80+.
- **Shapes**: Plain `nn.Linear` projections with `in_features` and `out_features` divisible by 16.
- **Implementation**: [`veomni/ops/torchao.py`](../../veomni/ops/torchao.py), invoked by [`veomni/ops/kernels/linear/standard/torchao.py`](../../veomni/ops/kernels/linear/standard/torchao.py) and [`veomni/ops/kernels/gated_mlp/standard/torchao.py`](../../veomni/ops/kernels/gated_mlp/standard/torchao.py).

---

## TorchAO MXFP8 (`dtype: mxfp8`)

Select MXFP8 on either dense op family. For example, this configuration runs
the attention projections and all three MLP projections in Qwen3 with MXFP8:

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: torchao
      kwargs: {dtype: mxfp8}
    gated_mlp_implementation:
      name: torchao
      kwargs: {dtype: mxfp8}
```

Native MXFP8 requires NVIDIA SM100+ and `torchao==0.18.0`. Set `emulated: true`
in a row's `kwargs` for emulation. The input and output feature dimensions must
both be multiples of 32. Token counts may be ragged or zero.

Each group of 32 elements shares an E8M0 scale for its E4M3 values. Forward and
input-gradient GEMMs quantize the supplied weight independently along their
reduction dimensions. The weight-gradient GEMM quantizes the input and output
gradient along the token dimension. The GEMMs produce BF16. At the model
boundary the projection returns the active autocast dtype, or the input dtype
when autocast is off. This also allows an FP32 model to enable MXFP8 on a single
attention projection. FSDP2 uses its ordinary floating-point AllGather by default.

On B200 with PyTorch 2.11, CUDA Inductor supports the linear row. With bias,
compiler fusion can round the sum differently from eager execution. For the
gated MLP, compiled down-projection weight gradients differed from an independent
reference by up to 6% in relative L2 in testing. Use eager execution or
`backend: aot_eager` for that row. CPU emulation works in eager mode; CPU
Inductor lacks the E8M0 scale lowering. See the
[compiler restrictions](../key_features/torch_compile.md).

### Optional FSDP2 AllGather

To quantize weights before the collective, add this setting alongside the
MXFP8 op selections above.

```yaml
model:
  accelerator:
    fsdp_config:
      mxfp8_all_gather: true
```

Only the dense weights consumed by a selected MXFP8 linear or gated MLP use
this transport. Biases, embeddings, the LM head and eager projections use
ordinary FSDP. The optimizer, distributed checkpoints and HF export see the
original master parameters. The frozen DPO reference uses ordinary AllGather.

Each rank rounds its local master shard to BF16 and independently quantizes
rows of the weight and its transpose. Forward uses the row layout, and the
input gradient uses the transposed layout. Both are gathered with their E8M0
scales. Together they carry 2.0625 bytes per weight element before collective
padding, compared with 2 bytes for BF16. This moves weight quantization onto
the local shard and avoids quantizing the gathered weight on each rank.

Use eager native FSDP2 with BF16 mixed precision and accelerator-resident
parameters. Each weight must be evenly sharded on dimension 0, with both
local dimensions divisible by 32. A shared projection weight must use MXFP8
with the same emulation setting at every alias. Unselected tied embeddings
and heads keep their ordinary FSDP behavior. Partial resharding and model compilation
are unsupported with this option. Per-site MXFP8 selections still apply.

---

## TorchAO FP4 QAT (`dtype: nvfp4` / `mxfp4`)

The same `torchao` row provides emulated FP4 quantization-aware training for
`linear` (`standard` only) and `gated_mlp` (`standard`). It borrows torchao's
tensor round trips to match deployment rounding and multiplies in the model
dtype (straight-through backward), so it does **not** require Blackwell hardware.

| `dtype` | Element | Scale | Group width (`block_size`) |
| --- | --- | --- | --- |
| `nvfp4` | E2M1 | E4M3 block scale + FP32 per-tensor scale (two-level dynamic `amax`) | 16 |
| `mxfp4` | E2M1 | E8M0 power-of-two block scale, no per-tensor scale | 32 |

For FP4 layouts, `emulated` defaults to `true`. **`emulated: false` for FP4 is
not implemented yet** and raises `NotImplementedError` rather than silently
falling back to emulation.

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: torchao
      kwargs: {dtype: nvfp4}   # or mxfp4
    gated_mlp_implementation:
      name: torchao
      kwargs: {dtype: nvfp4}
```

Start from [`configs/text/qwen3_nvfp4_qat.yaml`](../../configs/text/qwen3_nvfp4_qat.yaml),
set `model.model_path` and `data.train_path`, and run:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_nvfp4_qat.yaml
```

### FP4 QAT details and constraints

- **Operand dtypes**: Nonempty operands must be BF16 or FP32 (FP16 is rejected at runtime).
- **Group alignment**: Projections whose last dimension is not a multiple of the group width raise instead of padding.
- **Model coverage**: `torchao` is registered for `linear/standard` only. Models that also bind `linear/grouped` (such as DeepSeek-V4 `o_a_proj`) cannot select `linear_implementation: torchao` globally.
- **Activations**: Quantizing `gated_mlp` rows support `silu`, `swish`, `gelu`, and `gelu_pytorch_tanh` (with optional `swiglu_limit` clamping in model dtype) on the `qat` and `torchao` rows; `deepgemm` is narrower and serves only `silu` / `swish`, because it fuses the activation into its FP8 kernel. Unsupported activations raise instead of silently training unquantized.
- **Per-tensor `amax` (`nvfp4`)**: Computed over the rank-local activation tensor, so token packing, batch size, or sequence-parallel sharding can slightly shift the per-tensor scale.
- **Hardware**: CPU and CUDA SM80+ (native BF16 required for distributed CUDA training, enforced by `ops/qat/_hardware.reject_qat_without_native_bf16`). NPU and MLU reject `torchao` at config-parse time.
- **`torch.compile`**: `mxfp4` uses `float8_e8m0fnu` scales and is refused by `torch_compile`. `nvfp4` uses `float8_e4m3fn` scales (Inductor fails on pre-SM89 GPUs; on SM89 compiled vs eager can differ by ~0.5% of elements / ~1.2% relative L2, while SM120 is bit-identical).

---

## TileLang QAT and DeepGEMM MoE

For models with MoE experts and stored KV/indexer caches (such as DeepSeek-V4):

- **TileLang `qat`** (`linear`, `gated_mlp`, `moe_experts`) applies blockwise
  fake quantization (1x128 activation blocks, 128x128 FP8 weight tiles or
  1x32 FP4 expert groups) on NVIDIA SM90+ with `tilelang`.
- **`kv_round_implementation: triton`** rounds stored KV / indexer activations
  (SM89+); `none` disables rounding.
- **`moe_implementation: deepgemm`** runs **native FP8** grouped GEMMs
  (forward, dgrad, wgrad) on Hopper SM90 / Blackwell SM100. Today `deepgemm`
  supports `dtype: fp8` only; requesting another `dtype` raises `ValueError`.
  It composes cleanly with `qat` or `torchao` on `linear_implementation` and
  `gated_mlp_implementation`, because those are separate GEMMs on separate
  fields. The same `deepgemm` row is also available on `linear_implementation`
  (`standard` and `grouped`) and `gated_mlp_implementation` (`standard`), so the
  attention projections and the shared expert can run native FP8 too rather than
  being limited to fake quantization.

See [`configs/text/deepseek_v4_qat.yaml`](../../configs/text/deepseek_v4_qat.yaml)
for an all-QAT configuration and [FP8 MoE training](../design/deepgemm_moe.md)
for DeepGEMM requirements, `expert_fsdp_shard_dim`, and optional
`expert_weight_fp8_all_gather`.

---

## Per-Site Mixed-Precision `overrides`

Any of the four low-precision fields (`linear_implementation`,
`gated_mlp_implementation`, `moe_implementation`, `kv_round_implementation`)
accepts a mapping with `overrides` to run selected layers or projections on a
different row or kwargs (later rules win; `site` is matched with
`re.fullmatch`).

See [Choosing different rows per site](../design/op_selection.md#choosing-different-rows-per-site)
and the complete example in
[`configs/text/deepseek_v4_mixed_precision.yaml`](../../configs/text/deepseek_v4_mixed_precision.yaml).

---

## Distributed Behaviour, LoRA, and Checkpoints

- **FSDP2 / HSDP (decoder-layer wrapping)**: Supported. FSDP2 all-gathers each
  block's parameters before forward so quantizing rows see whole rank-local
  tensors. A raw `DTensor` operand is refused with `TypeError`.
- **`basic_modules: [Linear]` (per-projection wrapping)**: Not supported with
  `torchao` or `qat` (`FSDPLinear` subclasses cannot have their weights read
  directly); use `eager` (or per-site `overrides`) for those layers.
- **Sequence / context parallel & CPU offload**: Supported (see the `nvfp4`
  rank-local `amax` note above). Optional MXFP8 AllGather requires master
  shards on the accelerator.
- **LoRA**: Quantizing rows (`torchao`, `qat`, `deepgemm`) read base weights
  directly and refuse LoRA-wrapped projections (`LoraLinear` / MoE-LoRA) rather
  than silently bypassing adapters. Use `eager` (or `liger_kernel` for
  `gated_mlp`, `fused_triton` for MoE) on adapted layers, or exclude them via
  per-site `overrides`.
- **Checkpoints & resume**: Weight names and formats are unchanged, so DCP
  resume and HF safetensors export/reload work out of the box (exported weights
  are standard floating-point, not packed FP8/FP4). **Always re-specify your
  `ops_implementation` fields on resume**, as the trainer's `OpsImplementationConfig`
  takes precedence over the checkpoint's `config.json`.

---

## Tests and CI Coverage

Config round-tripping, per-site handle resolution, lazy-import guards,
shape/recipe validation, hardware checks, and missing/mismatched-torchao errors
run on CPU without `torchao`. GPU tests (native `aten::_scaled_mm`, FSDP2
training, DCP resume, HF export) require the `float8` extra and skip when
`torchao` is absent:

```bash
uv sync --locked --extra gpu --extra float8 --dev
pytest tests/models/test_float8_training.py tests/models/test_qat_fsdp.py tests/models/test_mxfp8_fsdp.py \
  tests/ops/linear/ tests/ops/gated_mlp/ tests/models/transformers/qwen/test_qwen3.py -q
```
