# Dense Qwen3 MXFP4 QAT

`torchao_mxfp4` fake-quantizes the operands of selected projections to MXFP4 so
training sees the rounding an MXFP4 deployment kernel applies. It is an op
implementation, selected exactly like `liger_kernel` or `eager` — not a separate
quantization recipe, and not a module rewrite. Parameters stay ordinary floating
point tensors with unchanged names, and checkpoint/export paths are untouched.

The recipe uses torchao's **emulated** MXFP4: the quantize-dequantize round trip
comes from torchao's own `MXTensor`, and the GEMM runs in the model dtype. No
native FP4 training kernels and no Blackwell requirement.

## The two op rows

| Field | Op | What it covers |
| --- | --- | --- |
| `linear_implementation` | `linear` / `standard` | A model that owns its projections individually (Qwen3 attention `q/k/v/o`) |
| `swiglu_mlp_implementation` | `swiglu_mlp` / `standard` | A model that runs the fused SwiGLU row (Qwen3 MLP `gate/up/down`) |

They are independent, and they are two of the four selections that make up a
quantization-aware training recipe — the others being `moe_implementation` and
`kv_round_implementation` for models that have experts or a KV cache. A model
that fuses its MLP reads only the second field; a model with per-projection
control reads the first. Set both to quantize every projection. There is no
global recipe switch: each family owns its own row. See
[op selection](../design/op_selection.md).

## Recipe

| Property | Value |
| --- | --- |
| Format | MXFP4: E2M1 element, E8M0 (power-of-two) block scale |
| Group width | 32, the OCP MX width |
| Scale calculation | RCEIL |
| Quantized | Both GEMM operands (activation and weight) |
| Backward | Straight-through, using the quantized operands |
| Dependency | `torchao`, optional; the row is unavailable without it |
| Device | Device-agnostic PyTorch math. Validated on CPU and NVIDIA CUDA; NPU and MLU are untested |

A projection whose last dimension is not a multiple of 32 raises rather than
being padded: the deployment kernels index one scale per 32-wide group, so a
partial group has no defined scale.

## Configure training

`torchao` is not part of any VeOmni extra. Install it alongside the GPU extra:

```bash
uv sync --locked --extra gpu --dev
uv pip install torchao==0.18.0
```

Select the implementation under the model's existing ops config:

```yaml
model:
  ops_implementation:
    linear_implementation: torchao_mxfp4
    swiglu_mlp_implementation: torchao_mxfp4
```

An unselected field resolves to `eager`, so a config that sets only
`linear_implementation` trains a weight+activation MXFP4 attention with an
unquantized MLP. Selecting `torchao_mxfp4` without `torchao` installed raises
`Op 'linear' variant='standard' impl='torchao_mxfp4' requires unavailable
package(s): 'torchao'` at model construction; it never falls back silently.

### Not supported: LoRA on a quantized projection

A `linear` row fake-quantizes the projection's own `weight`. LoRA replaces the
matched `nn.Linear` with a `LoraLinear` that keeps the base layer at
`base_layer`, so the weight is no longer reachable as `q_proj.weight`. A model
built with `linear_implementation: torchao_mxfp4` therefore raises
`cannot run on a wrapped projection` as soon as an adapter on those projections
forwards, instead of quietly training the unquantized base layer while the config
still claims MXFP4.

Set `linear_implementation: eager` for adapter training. The shipped
[Qwen3 LoRA config](../../configs/text/qwen3_lora.yaml) also pins
`swiglu_mlp_implementation: eager`, as do the other text LoRA configs that
target MLP projections.

The fused MLP row follows the same rule, and only for the quantizing value:
with `swiglu_mlp_implementation: torchao_mxfp4` and an adapter on
`gate_proj`/`up_proj`/`down_proj`, the forward raises
`swiglu_mlp_implementation=... cannot run on a wrapped projection`.
`liger_kernel` is **not** affected — it is an arithmetic rewrite that keeps the
projection-module fallback, so leaving `swiglu_mlp_implementation` unset under
LoRA still resolves to `liger_kernel` and still works. Only a fake-quant row
needs the projection's own `weight`.

Start from [the Qwen3 example](../../configs/text/qwen3_mxfp4_qat.yaml), set
`model.model_path` and `data.train_path`, then run:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_mxfp4_qat.yaml
```

The example's learning rate and duration are starting settings, not a validated
convergence recipe.

## Distributed behaviour

The fake quantization runs inside the op, on whichever tensor the caller passes,
and it is per-row: MX groups span the last dimension, so a row split is never a
group split. Every placement the parallel plans produce quantizes the same
values it would unsharded:

| Operand | Result |
| --- | --- |
| Plain tensor (FSDP2 after the all-gather) | exact |
| `Replicate` DTensor | exact; the local shard is the whole tensor |
| Shard along the output dimension | exact; each shard copies whole groups |
| Shard along the quantized dimension, width a multiple of 32 | exact |
| Shard along the quantized dimension, width not a multiple of 32 | `ValueError` |

The last row is the only unsupported case, and it fails rather than rounding
wrongly: a partial group has no defined scale. It is caught by the operand's own
length, not by inspecting the placement, because the placement is not what
determines correctness.

Nothing here is specific to a parallelism degree, so FSDP2, HSDP, sequence or
context parallel, tensor parallel and expert parallel are all permitted. **What
has not been measured is convergence under them** — the numerical validation is
single-device. Treat multi-rank QAT as untested, not as unsupported.

BF16 parameters with FP32 gradient reduction is what the example config pins and
what was validated. `torch.compile` is the one combination refused outright, at
config time, per the next subsection.

### `torch.compile` is rejected

The MXFP4 row materializes `float8_e8m0fnu` scale tensors, and Inductor has no
lowering for that dtype — the failure is
`AssertionError: torch.float8_e8m0fnu missing from torch._inductor.codegen.cpp.DTYPE_TO_CPP`,
raised from inside a fused kernel. `compile_decoder_blocks()` therefore rejects
`linear_implementation: torchao_mxfp4` and `swiglu_mlp_implementation: torchao_mxfp4`
before compiling, so the error names the config field instead of the assertion.

Offloading is not restricted: CPU offload moves the shard to host and the op sees
an ordinary tensor, so the fake quantization is unaffected.

## Checkpoints and export

Weight names are unchanged, so DCP resume, HF safetensors export and reload all
work with no extra state: the fake-quant scales are derived on every forward and
never persisted. An exported checkpoint holds ordinary FP32/BF16 floating-point
weights, not packed MXFP4 tensors — producing a deployable MXFP4 checkpoint is
separate work.

The selection lives in `config.json` under `ops_implementation`, so a reloaded
model keeps training with the same rounding.

## Tests

```bash
uv pip install torchao==0.18.0   # not in any extra; the rows are optional
pytest tests/ops/linear/ tests/ops/swiglu_mlp/ -q
```

These are CPU tests: they cover the registry rows, forward parity against a
hand-quantized reference, the straight-through gradient contract, the
empty-input and non-contiguous-input paths, the divisibility and dtype errors,
and the missing-package failure. `tests/models/transformers/qwen/test_qwen3.py`
additionally checks that a built Qwen3 model runs the selected row, produces
finite gradients, and refuses a LoRA-wrapped projection.

**They skip when `torchao` is absent**, and `torchao` is deliberately in no
VeOmni extra — the `gpu` extra is the accelerator stack, and a fake-quant
backend should not add a wheel to every GPU job. CI therefore reports them as
skipped, not passed: install `torchao` to run them, and treat a change that
removes `torchao` from a test environment as reducing what is verified.
