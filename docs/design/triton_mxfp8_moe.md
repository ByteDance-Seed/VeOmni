# Triton MXFP8 MoE training

Select the existing fused expert implementation in your training YAML:

```yaml
model:
  ops_implementation:
    moe_implementation: fused_triton_fp8
```

This registers `fused_triton_fp8` for the standard `moe_experts` VeomniOp. It requires Triton with
`tl.dot_scaled`, a Blackwell SM10x CUDA GPU (B200/GB300), BF16 inputs and
weights, merged `gate_up_proj` weights, SiLU/SwiGLU, and EP1. The forward/dgrad reduction
dimensions must be positive multiples of 128. Split expert weights and
expert parallelism raise an error rather than silently falling back.

Selecting `fused_triton_fp8` always quantizes both gate/up and down
projections in forward, dgrad, and wgrad. There are no projection or
forward-only quantization modes.

The router, scatter/gather, SwiGLU, routing-weight multiplication, and their
backward arithmetic follow the existing fused Triton implementation.
Use `fused_triton` for the BF16 baseline.

Each 32-element reduction block uses E4M3 values and an E8M0 power-of-two
scale. The exponent is `ceil(log2(max(abs(x)) / 448))`, bounded to
`[-126, 127]`, with the existing small-value floor. GEMMs use
`tl.dot_scaled`, FP32 accumulation, and BF16 outputs. Weight gradients
quantize transposed inputs independently per expert, pad partial 128-token
tiles with zeros, and return zero gradients for empty experts.

Quantization runs on each invocation; there is no persistent quantized
weight cache or checkpoint format change. Kernel operands and saved expert activations
remain BF16; optimizer/master-parameter dtype follows the training config. MXFP8 backward is approximate, so validate training quality for
your workload. Host scheduling synchronizes expert offsets to validate
boundaries and determine the launch size; this implementation does not
support CUDA graph capture or full-graph torch.compile.

Run dispatch/validation tests on CPU and numerical tests on SM10x:

```bash
pytest tests/ops/moe_experts/test_triton_mxfp8.py -v
```

The port uses the current upstream fused Triton scatter/gather, activation,
and autograd path. Routing weights multiply the down-projection output,
as in upstream `fused_triton`; the original internal MR applied them before
the down projection. Quantization is nonlinear, so the two versions need
separate workload quality measurements. The four quantization/GEMM kernels
retain the internal recipe unchanged.
