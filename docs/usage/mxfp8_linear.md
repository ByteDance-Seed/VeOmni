# Native MXFP8 linear projections

Install the optional torchao dependency with `uv sync --extra gpu --extra float8 --dev`.
Run dense Qwen3 with FSDP2 on at least two NVIDIA SM100+ GPUs. Choose native
arithmetic at each projection through the linear op selection. This example
uses the three MLP projections in each decoder layer.

```yaml
model:
  ops_implementation:
    linear_implementation:
      name: eager
      overrides:
        - site: 'model\.layers\.\d+\.mlp\.(gate|up|down)_proj'
          name: torchao
          kwargs: {dtype: mxfp8, emulated: false}
  accelerator:
    fsdp_config:
      mxfp8_all_gather: false
```

Keep BF16 parameter communication and FP32 gradient reductions. With
`mxfp8_all_gather: true`, FSDP2 quantizes selected FP32 weight shards
before gathering them. The two orientations and their scales cost 2.0625 bytes
per parameter. Input dimensions and local output rows must align to 32.
Native MXFP8 currently requires ordinary FSDP2 without compilation, adapters,
extra parallel dimensions or offload.

Both transport modes export ordinary floating-point weights. A training job
passes its own ops configuration to the loader. A fresh job can use the eager
linear row on other hardware. For an exact DCP continuation, pass the original
row selection and transport setting again. Loading a saved ops config directly
requests its recorded kernels. Upstream HF can read the floating-point weights.

The native test selection runs the operator oracle and three two-rank
training tests, including HF reload and fresh-process DCP continuation.
Configure `MXFP8_SM100_RUNNER_LABEL` with a self-hosted runner that exposes two
SM100+ devices to the CI container.
