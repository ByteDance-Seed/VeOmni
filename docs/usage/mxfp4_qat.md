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

Together they cover the **attention projections and the MLP projections**.
`lm_head` is excluded: the causal-LM path either calls it directly or hands its
weight to the loss operator, so neither row is on its call path — do not read
these selections as producing a quantized output head.

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

An unselected field falls back to that family's host default — `eager` for
`linear`, `liger_kernel` on GPU/MLU and `eager` on NPU for `swiglu_mlp`. Either
way the MLP is **not** quantized, so a config that sets only
`linear_implementation` trains MXFP4 attention with a full-precision MLP. Selecting `torchao_mxfp4` without `torchao` installed raises
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

These rows are **tensor-level**: they round the operand they are handed and pass
it to `F.linear`. They do not issue collectives, so they can only run on an
operand that is already whole on the calling rank.

| Operand | Result |
| --- | --- |
| Plain tensor | works |
| `Replicate` DTensor | works; the local shard is the whole tensor |
| Sharded DTensor (`Shard(0)`, `Shard(1)`, …) | `TypeError` |

A sharded weight is refused rather than mis-multiplied. `Shard(0)` leaves partial
*output* features, needing a concat afterwards; `Shard(1)` leaves a partial
*input* whose product needs an all-reduce. The per-shard arithmetic is fine, but
returning it would present a partial product as a complete one.

This does not restrict FSDP2: a fully sharded FSDP2 mesh all-gathers each block's
parameters before its forward, so the row sees a whole tensor. What it does rule
out is keeping a weight sharded *through* the GEMM, which is tensor parallelism —
and VeOmni does not implement that anyway (`AcceleratorConfig` asserts
`tp_size == 1`).

| Mode | Status |
| --- | --- |
| FSDP2 / HSDP sharding | works via FSDP2's own all-gather |
| Sequence / context parallel | works; the sharded axis is the sequence |
| Expert parallel | works; expert weights reach the GEMM whole on the owning rank |
| Tensor parallel | not implemented in VeOmni, independently of this row |

**Convergence under any multi-rank setting is unmeasured** — the numerical
validation is single-device. Treat multi-rank QAT as untested, not as validated.

`torch.compile` is refused outright, at config time, per the next subsection.

### `torch.compile` is rejected

The MXFP4 row materializes `float8_e8m0fnu` scale tensors, and Inductor has no
lowering for that dtype — the failure is
`AssertionError: torch.float8_e8m0fnu missing from torch._inductor.codegen.cpp.DTYPE_TO_CPP`,
raised from inside a fused kernel. `compile_decoder_blocks()` therefore rejects
`linear_implementation: torchao_mxfp4` and `swiglu_mlp_implementation: torchao_mxfp4`
before compiling, so the error names the config field instead of the assertion.

Offloading is not restricted: CPU offload moves the shard to host and the row sees
an ordinary tensor.

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
