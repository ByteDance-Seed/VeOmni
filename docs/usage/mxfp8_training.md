# Dense Qwen3 MXFP8 training

MXFP8 uses native torchao matrix multiplication for selected dense Qwen3
projections in forward, input-gradient and weight-gradient computation. Weights
and activations use E4M3 payloads with one E8M0 scale per 32 values (RCEIL).
The FSDP2 optimizer retains FP32 master shards; computation-facing parameters
and outputs are BF16 and gradient reduction is FP32.

This recipe requires `torchao==0.18.0` and at least two NVIDIA SM100+ GPUs
for VeOmni's FSDP2 training path. Each rank
checks its current device and executes a native GEMM probe before conversion
or sharding. An unsupported CUDA/cuBLAS build fails explicitly. Training never
falls back to CPU emulation.

## Configure training

Install the normal GPU environment with `uv sync --locked --extra gpu --dev`.
Start from [the Qwen3 example](../../configs/text/qwen3_mxfp8.yaml) and set the
model and prepared conversation dataset paths:

```yaml
model:
  accelerator:
    fsdp_config:
      mixed_precision:
        mxfp8:
          enable: true
          enable_fsdp_all_gather: false
          target_modules:
            - model.layers.*.mlp.gate_proj
            - model.layers.*.mlp.up_proj
            - model.layers.*.mlp.down_proj
```

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_mxfp8.yaml
```

The launcher uses all visible GPUs. `global_batch_size: null` derives the batch
as `micro_batch_size × data_parallel_size`; four and eight ranks therefore work
without changing a hard-coded two-rank batch. The learning rate and duration
are starting settings, not a validated convergence recipe.

Targets are complete module-name glob patterns. Each pattern must match;
unsupported matches raise rather than silently keeping BF16. Supported targets
are bias-free `model.layers.<index>.mlp.{gate,up,down}_proj` and
`model.layers.<index>.self_attn.{q,k,v,o}_proj`. Both dimensions must be divisible
by 32. Selected weights must not be shared. The LM head is excluded because a
fused loss may read its weight without calling its forward.

The initial scope is standalone dense Qwen3 with ordinary FSDP2 and AdamW.
HSDP, SP/CP, TP/PP, ExtraParallel, asynchronous attention, LoRA, parameter or
activation offload, custom parallelization, QAT and compilation are unsupported.
Other low-precision communication options cannot be combined with this recipe.
No trainer or optimizer-step lifecycle hooks are required: MX block scales are
derived from the current weight on each use.

## Optional parameter AllGather

With `enable_fsdp_all_gather: false`, FSDP2 gathers ordinary BF16 parameters.
With it enabled, each FP32 local shard is quantized separately in two
orientations for forward and input-gradient GEMMs. Both orientations are
derived directly from the master. The backward orientation is never constructed
by re-quantizing the already lossy forward orientation.

Quantized AllGather requires uniform dim-0 shards whose row counts are
multiples of 32. This preserves the block partition used by full-weight
quantization. Incompatible projection sizes fail before FSDP2 wrapping.

The transport sends two E4M3 payloads and their scales, totaling **2.0625 bytes
per weight** before collective overhead, versus BF16's 2 bytes. This establishes
the MXFP8 execution contract; it is not a communication-volume reduction.
AllGather on/off also differ in weight rounding: the BF16 transport path rounds
before quantization, while the MX path quantizes FP32 masters directly. They
should not be expected to match bitwise.

## Loading, checkpoints and export

The [parallel builder](../../veomni/distributed/torch_parallelize.py#L1030)
[validates the recipe](../../veomni/distributed/mxfp8/__init__.py#L77),
[converts projections](../../veomni/distributed/mxfp8/__init__.py#L219) after
FP32 upcast, then wraps and materializes them through the existing FSDP2/HF loader.
SwiGLU calls converted projection forwards instead of reading their weights
through a fused BF16 path. Parameter names stay unchanged.

DCP stores plain FP32 master tensors and optimizer state. Rebuild the same
recipe and target selection when resuming DCP. Transport payloads and block
scales are derived state and are not checkpointed. Exporting a gathered compute
view is rejected; the model must be resharded to expose its master weights.

Ordinary HF export writes floating-point weights, normally rounded to BF16 by
the existing exporter. It does not produce a packed deployment checkpoint or
preserve optimizer state. The HF config includes `veomni_mxfp8` recipe metadata.
The VeOmni parallel builder preserves this recipe when
`mixed_precision.mxfp8` is unset, including through the trainer's default config.
An explicit recipe replaces the saved one; `mxfp8: {enable: false}` disables it
and records that choice for subsequent exports. Loading with an upstream HF
inference runtime does not enable MXFP8 by itself.

## Validation

[CPU tests](../../tests/ops/test_mxfp8_tensors.py) check independent complete-weight
quantization against gathered shards, both weight orientations, ragged-token
forward/backward, master I/O, fused/foreach AdamW and CPU FSDP2/DCP continuation.
[Configuration tests](../../tests/models/test_mxfp8_config.py) cover YAML/CLI,
saved recipes, explicit disable, rank-local preflight, unsupported combinations,
projection dispatch and example batches across DP sizes.
[CPU loading tests](../../tests/models/test_mxfp8_loading.py) use generated Qwen3,
two Gloo ranks and emulated GEMMs to exercise the public meta/HF loader,
non-reentrant checkpointing, production fused AdamW, HF export rounding and
recipe inheritance/disable. They do not substitute for native kernel validation.

Run [native acceptance](../../tests/models/test_mxfp8_training.py) on two
SM100+ devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 pytest tests/models/test_mxfp8_training.py -q
```

These tests are selected by the existing `tests/models/` and `tests/ops/` CI
directory entries. Native cases skip on unsupported hardware, including L20.
A test definition or CPU pass does not establish a native hardware result.
Native multi-GPU acceptance, representative convergence, throughput and memory
benefits require separate measurements on the supported hardware.
