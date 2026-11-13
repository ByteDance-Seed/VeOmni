# Dense Qwen3 torchao FP4 QAT

This page documents the torchao-backed FP4 quantization-aware training row,
`torchao_nvfp4`.

`torchao_nvfp4` fake-quantizes the operands of selected projections to NVFP4 so
training sees the rounding an NVFP4 deployment kernel applies with dynamic
two-level runtime scaling (not a calibrated static activation scale). It is an
op implementation, selected exactly like `liger_kernel` or `eager` — not a
separate quantization recipe, and not a module rewrite. Parameters stay ordinary
floating point tensors with unchanged names, and checkpoint/export paths are
untouched.

The row uses torchao's **emulated** NVFP4 round trip (`NVFP4Tensor`), and the
GEMM runs in the model dtype (nonempty operands must be BF16 or FP32; FP16
operands are rejected at runtime even though `mixed_precision.param_dtype`
accepts `float16`). No native FP4 training kernels and no Blackwell requirement
during QAT.

## The two op rows

| Field | Op | What it covers |
| --- | --- | --- |
| `linear_implementation` | `linear` / `standard` | A model that owns its projections individually (Qwen3 attention `q/k/v/o`) |
| `swiglu_mlp_implementation` | `swiglu_mlp` / `standard` | A model that runs the fused SwiGLU row (Qwen3 MLP `gate/up/down`) |

Together they cover the **attention projections and the MLP projections**.
`lm_head` is excluded: the causal-LM path either calls it directly or hands its
weight to the loss operator, so neither row is on its call path — do not read
these selections as producing a quantized output head.

They are independent, and they are two of the four selections that make up a
quantization-aware training recipe — the others being `moe_implementation` and
`kv_round_implementation` for models that have experts or a KV cache. There is
no global recipe switch: each family owns its own row. See
[op selection](../design/op_selection.md).

`torchao_nvfp4` is registered for `linear` / `standard` only. DeepSeek-V4 also
binds `linear` / `grouped` (its `o_a_proj`), so it cannot select
`linear_implementation: torchao_nvfp4`; model construction raises instead of
quantizing only some of its projections.

## Recipe

| Property | `torchao_nvfp4` |
| --- | --- |
| Format | NVFP4: E2M1 element, E4M3 block scale + FP32 per-tensor scale |
| Group width | 16, the NVIDIA NVFP4 width (`block_size` other than 16 raises) |
| Scale calculation | Two-level dynamic `amax` (`NVFP4Tensor`) |
| Operand dtypes | BF16 or FP32 (FP16 rejected at runtime) |
| Quantized | Both GEMM operands (activation and weight) |
| Backward | Straight-through, using the quantized operands |
| Dependency | `torchao`, optional; the row is unavailable without it. Validated on `torchao==0.18.0` |
| Device | Device-agnostic PyTorch math. CPU and NVIDIA CUDA are supported. **NPU and MLU are rejected at config-parse time** (`arguments_types._NPU_ALLOWED` / `_MLU_ALLOWED` exclude the row), not merely untested |

The row uses torchao's prototype `NVFP4Tensor` API. The registry only checks
that `torchao` is installed, so an older torchao without that API (for
example the ROCm image's `0.9.0`) reports the row as available and then raises
`ImportError` at the first forward.

A projection whose last dimension is not a multiple of 16 raises rather than
being padded: the deployment kernels index one scale per group, so a partial
group has no defined scale.

The SwiGLU row quantizes the input once for `gate` and `up`, and applies the
optional `swiglu_limit` clamp and `silu * up` in the model dtype on the
dequantized projection outputs, matching the TileLang `qat` row: a deployment
fused kernel clamps BF16 projection outputs, not FP32 ones.

The per-tensor activation scale is computed from the dynamic `amax` over the
entire activation tensor on the calling rank. Changing token packing, batch
size, or sequence-parallel partitioning can therefore change the activation
scale and rounding for a given token compared with unsharded training or
inference deployment.

## Configure training

`torchao` is not part of any VeOmni extra. Install it alongside the GPU extra:

```bash
uv sync --locked --extra gpu --dev
source .venv/bin/activate      # train.sh resolves torchrun from PATH
uv pip install torchao==0.18.0
```

Select the implementation under the model's existing ops config:

```yaml
model:
  ops_implementation:
    linear_implementation: torchao_nvfp4
    swiglu_mlp_implementation: torchao_nvfp4
```

An unselected field falls back to that family's host default — `eager` for
`linear`, `liger_kernel` on GPU/MLU and `eager` on NPU for `swiglu_mlp`. Either
way that family is **not** quantized, so a config that sets only
`linear_implementation` trains NVFP4 attention with a full-precision MLP.
Selecting `torchao_nvfp4` without `torchao` installed raises
`Op 'linear' variant='standard' impl='torchao_nvfp4' requires unavailable
package(s): 'torchao'` at model construction; it never falls back silently.
A quantizing `swiglu_mlp` row on a model whose activation is not SiLU is
refused at config load, as for every quantizing `swiglu_mlp` row.

Start from [the Qwen3 example](../../configs/text/qwen3_nvfp4_qat.yaml), set
`model.model_path` and `data.train_path`, then run:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_nvfp4_qat.yaml
```

The example's learning rate and duration are starting settings, not a validated
convergence recipe.

### Not supported: LoRA on a quantized projection

A quantizing row fake-quantizes the projection's own `weight`. LoRA replaces the
matched `nn.Linear` with a `LoraLinear` that keeps the base layer at
`base_layer`, so the weight is no longer reachable as `q_proj.weight`. A model
built with `linear_implementation: torchao_nvfp4` therefore raises
`linear_implementation='torchao_nvfp4' cannot run on 'LoraLinear'` as soon as an
adapter on those projections forwards, instead of quietly training the
unquantized base layer while the config still claims NVFP4.

Set `linear_implementation: eager` for adapter training: the eager row calls the
projection module, so the adapter's own forward runs. The fused MLP row follows
the same rule, and only for the quantizing value: with
`swiglu_mlp_implementation: torchao_nvfp4` and an adapter on
`gate_proj`/`up_proj`/`down_proj`, the forward raises
`swiglu_mlp_implementation='torchao_nvfp4' cannot run on a wrapped projection`.
`liger_kernel` is **not** affected — it is an arithmetic rewrite that keeps the
projection-module fallback.

## Distributed behaviour

The rows are **tensor-level**: they round the operands they are handed and pass
them to `F.linear`. They do not issue collectives, so they can only run on an
operand that is already whole on the calling rank. A `DTensor` operand is
refused with `TypeError`, the same contract as the TileLang FP8 primitives: a
`DTensor` reports its global shape, so the group-width check would validate a
shape the rank does not hold, and the per-tensor `amax` would depend on the
partitioning.

| Mode | Status |
| --- | --- |
| FSDP2 / HSDP sharding (decoder-layer wrapping) | works: FSDP2 all-gathers each block's parameters before its forward, so the row sees whole tensors |
| `basic_modules: [Linear]` (per-projection FSDP2 wrapping) | `eager` works, because it calls the projection module and its unshard hook runs; `torchao_nvfp4` reads the weight directly and refuses the still-sharded `DTensor` |
| Sequence / context parallel | works; the sharded axis is the sequence (see the `amax` note above) |
| Tensor parallel | not implemented in VeOmni, independently of this row |

**Convergence under any multi-rank setting is unmeasured** — the numerical
validation is single-device. The example config pins BF16 parameters with FP32
gradient reduction, but that is a **configuration choice, not a validated
result**.

Offloading is not restricted: CPU offload moves the shard to host and the row sees
an ordinary tensor.

### `torch.compile`

`torchao_nvfp4` uses `float8_e4m3fn` block scales and an FP32 per-tensor scale.
Inductor compilation can fail on pre-SM89 CUDA GPUs because Triton does not
support the E4M3 casts there; disable `torch.compile` on those GPUs when using
`torchao_nvfp4`.

Whether compiled rounding matches the eager round trip depends on the GPU.
`torch.compile(nvfp4_fake_quant)` was measured on a `4096 x 1024` BF16 tensor:

| GPU | Compiled vs eager |
| --- | --- |
| RTX PRO 6000 Blackwell (SM120) | bit-identical |
| RTX 4060 Laptop (SM89) | about 0.5% of elements on a neighbouring FP4 code (relative L2 about 1.2%) |

The SM89 difference most likely comes from Inductor fusing the two-level scale
division differently from torchao's eager kernels. The eager path is the one
the op tests pin, so treat a compiled NVFP4 run on any GPU other than SM120 as
unvalidated against that reference.

## Checkpoints and export

**A restart must re-select the rows.** The trainer passes its own
`OpsImplementationConfig` when it builds the model, and that object replaces
whatever the exported checkpoint recorded. Launching a resumed run without
`linear_implementation` / `swiglu_mlp_implementation` therefore trains at full
precision even though the checkpoint's `config.json` still lists the rows.

Weight names are unchanged, so DCP resume, HF safetensors export and reload all
work with no extra state: the fake-quant scales are derived on every forward and
never persisted. An exported checkpoint holds ordinary FP32/BF16 floating-point
weights, not packed NVFP4 tensors — producing a deployable NVFP4 checkpoint is
separate work.

## Tests

```bash
uv pip install torchao==0.18.0   # not in any extra; the rows are optional
pytest tests/ops/linear/ tests/ops/swiglu_mlp/ tests/models/transformers/qwen/test_qwen3.py -q
```

The op tests cover the registry rows, forward parity against torchao's own
NVFP4 round trip, the straight-through gradient contract, the empty-input path,
and the group-width and dtype errors.
`tests/models/transformers/qwen/test_qwen3.py` additionally checks that a built
Qwen3 model binds the selected row, that the `eager` row calls each projection
module, that NVFP4 training produces finite gradients (CPU, and CUDA when
available), and that a LoRA-wrapped projection is refused.

**They skip when `torchao` is absent**, and `torchao` is in no VeOmni extra.
CI therefore reports them as skipped, not passed: install `torchao` to run them,
and treat a change that removes `torchao` from a test environment as reducing
what is verified.
