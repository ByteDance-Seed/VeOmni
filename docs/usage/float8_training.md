# Tensorwise float8 training

VeOmni can opt selected dense `torch.nn.Linear` modules into torchao's dynamic
tensorwise float8 forward and backward GEMMs. Parameters and optimizer updates
remain FP32; FSDP2 unshards parameters to BF16 and reduces gradients in FP32.
The feature is disabled by default and does not import torchao when disabled.

This is a first compute-only slice of [RFC #1360](https://github.com/ByteDance-Seed/VeOmni/issues/1360).
It does not enable float8 parameter AllGather, MXFP8, QAT, or a different
gradient ReduceScatter policy. Existing `qat_implementation` behavior is unchanged.

## Install and configure

Install the optional GPU environment, which includes the pinned torchao backend:

```bash
uv sync --extra gpu --dev
```

The `gpu` extra pins `torchao==0.18.0`. The GPU environment retains its
existing PyTorch `2.11.0+cu130` and Transformers `5.16.1` pins.

For dense Qwen3, select the MLP projections explicitly:

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
          target_modules:
            - model.layers.*.mlp.gate_proj
            - model.layers.*.mlp.up_proj
            - model.layers.*.mlp.down_proj
```

Patterns match full module names. `*` matches any characters, following
VeOmni's existing module-name matching. Every configured pattern must match.
Eligible names are `model.layers.<index>.mlp.{gate,up,down}_proj` and
`model.layers.<index>.self_attn.{q,k,v,o}_proj`. The LM head is rejected because
VeOmni's training loss reads its weight directly instead of calling its forward.
Matched modules must be plain `torch.nn.Linear` instances; subclasses,
LoRA-wrapped modules, tied/shared selected parameters, and dimensions not
divisible by 16 raise an error before conversion. Unmatched modules, including
the LM head in this example, retain their existing computation.

The recipe enables torchao's native inner-dimension zero padding. Flattened
token counts such as 17 or 33 therefore work for weight-gradient GEMMs without
changing output shapes or using emulation. Padding can add temporary allocations.

## Supported initial surface

| Property | Initial support |
|---|---|
| Models | Dense Qwen3 and Llama |
| Modules | Explicitly selected plain Linear modules, dimensions divisible by 16 |
| Hardware | NVIDIA CUDA SM89 or newer; validated on RTX 4090 |
| Recipe | Dynamic tensorwise float8, native GEMMs |
| Parameters / optimizer | FP32 master parameters and standard optimizer updates |
| FSDP computation / reduction | BF16 / FP32 |
| Parallelism | FSDP2 with one data-parallel shard mesh; no HSDP replication |
| Checkpoints | Existing HF weight loading, DCP model/optimizer resume, and floating-point HF export |

DDP, HSDP replication, sequence/context/tensor/pipeline/ExtraParallel,
FSDP CPU offload, LoRA/PEFT, and `torch.compile` are rejected by this initial
integration. Other model families are rejected even if some of their Linear
modules could be converted in isolation. This avoids implying support for
model-specific lifecycles that have not been checked.

The existing trainer requires more than one rank for its FSDP2 path. A direct
one-rank `parallelize_model_fsdp2` test checks local lifecycle behavior, but does
not establish multi-card collective correctness or end-to-end trainer throughput.

## Initialization and checkpoint contract

The framework owns conversion; models need no new hook or method. Configuration
is defined by [Float8TrainingConfig](../../veomni/arguments/arguments_types.py#L547).

1. The model is built on meta with FP32 parameters.
2. [build_parallelize_model](../../veomni/distributed/torch_parallelize.py#L1010)
   validates the runtime and retains the existing
   FP32 master-parameter policy.
3. [parallelize_model_fsdp2](../../veomni/distributed/torch_parallelize.py#L426)
   calls [convert_float8_linears](../../veomni/distributed/float8.py#L68) at
   the [pre-wrap conversion site](../../veomni/distributed/torch_parallelize.py#L470),
   which validates every target and converts the selected
   Linear modules before wrapping transformer blocks and the root with FSDP2.
4. The existing materialization/weight loader writes the original weight/bias
   keys. On full DCP resume, initial HF loading is skipped as before.
5. FSDP2 supplies BF16 unsharded parameters during forward/backward. Torchao
   derives tensorwise scales from the current operands on every operation.
6. The optimizer updates the FP32 parameter shards. A following operation
   derives new float8 values/scales from those updated parameters.

VeOmni's Qwen3/Llama MLP normally uses a weight-only fused SwiGLU path. Its
[plain-projection check](../../veomni/ops/kernels/swiglu_mlp/__init__.py#L35)
requires exact `nn.Linear` instances: converted
`Float8Linear` subclasses take the projection-module forward path, preserving
the float8 computation. Attention projections already call their modules.

With float8 AllGather disabled, torchao retains ordinary master parameters and
adds no persistent quantization buffers. The parameter names and checkpoint
layout stay unchanged. DCP persists model and optimizer state through the
existing checkpointer. An HF export contains the usual floating-point model
weights, not an inference-quantized deployment format. To resume float8 training,
enable the same float8 configuration before loading the checkpoint; enabling or
disabling float8 changes compute numerics even though the weight keys match.

## Validation and limits

`tests/models/test_float8_training.py` includes CPU configuration, selection,
unsupported-combination and disabled-dependency tests. Native GPU cases check
actual `aten::_scaled_mm` execution, BF16 output/input-gradient/weight-gradient
relative-error bounds, ragged token counts, meta conversion and HF loading,
FP32 optimizer updates, gradient accumulation, DCP model/optimizer resume, and
ordinary floating-point HF export. The existing models-directory selections
collect this file in both accelerator suite manifests. GPU unit CI installs the
`gpu` extra, including the pinned torchao backend; native cases
skip on unsupported hardware, including NPU. Local runs without the optional
dependency skip the native cases.

Run the native tests in the optional environment:

```bash
pytest tests/models/test_float8_training.py -q
```

These tests do not establish long-run convergence, an accuracy guarantee for
every tensor distribution, or a throughput/memory improvement. Float8 error
depends on the tensors being quantized, and eager scaling overhead can dominate
small models. Validate the intended workload against a fixed BF16 baseline
before relying on training efficiency or final model quality.
