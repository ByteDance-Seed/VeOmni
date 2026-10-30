# Dense Qwen3 MXFP4 QAT

`torchao_mxfp4` simulates MXFP4 activation and weight rounding in selected Dense
Qwen3 projections, using straight-through gradients and ordinary floating-point
parameters. This initial [low-precision RFC](https://github.com/ByteDance-Seed/VeOmni/issues/1360)
recipe uses torchao's **emulated** MXFP4, without native FP4 training GEMMs or a
Blackwell requirement. Training requires NVIDIA CUDA devices with native BF16
support (SM80 or newer).

## Recipe and scope

| Property | Initial recipe |
| --- | --- |
| Model | Standalone Dense Qwen3 |
| Targets | Explicit decoder attention or MLP projection module patterns |
| Quantization | MXFP4 E2M1 activations and weights, groups of 32, RCEIL scale calculation |
| Dependency | `torchao==0.18.0`, installed by the GPU extra |
| Training | NVIDIA CUDA, FSDP2, BF16 unsharded parameters, FP32 masters and gradient reduction |
| CPU | Distributed training unsupported |
| Parameter AllGather | Existing BF16 transport; no quantized AllGather |
| Checkpoints | DCP for exact optimizer continuation; ordinary HF floating-point export |

Unsupported: other model families, MoE, shared selected weights, biased
projections, and projection input widths not divisible by 32. `lm_head` is
excluded because training loss can read its weight without calling its forward.

Distributed use rejects DDP, HSDP, SP/CP, TP/PP, ExtraParallel, LoRA, FSDP parameter
CPU offload and `torch.compile`. Activation offload is unvalidated and disabled
in the example. Native DeepSeek `fp8_blockwise` QAT remains a separate recipe.

## Configure training

```bash
uv sync --locked --extra gpu --dev
```

The recipe and target selection belong to the model's existing ops config:

```yaml
model:
  ops_implementation:
    qat_implementation: torchao_mxfp4
    qat_target_modules:
      - model.layers.*.mlp.gate_proj
      - model.layers.*.mlp.up_proj
      - model.layers.*.mlp.down_proj
  accelerator:
    init_device: meta
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
        cast_forward_inputs: true
```

Patterns match complete module names; each must select a supported projection.
For attention, add `model.layers.*.self_attn.q_proj`, `k_proj`, `v_proj` and
`o_proj` as separate patterns with the same prefix. The default recipe is `none`.

Start from [the Qwen3 example](../../configs/text/qwen3_mxfp4_qat.yaml), set
`model.model_path` and `data.train_path` to your checkpoint and prepared conversation
dataset, then run on at least two CUDA devices:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_mxfp4_qat.yaml
```

By default, `train.sh` uses all visible GPUs. The example sets `global_batch_size: null`
to derive it as `micro_batch_size × data_parallel_size`, with one microbatch per step.
With `micro_batch_size: 1`, four and eight GPUs use global batches of four and eight.

The example's learning rate and duration are starting settings, not a validated convergence recipe.

## Construction and checkpoints

[The model builder](../../veomni/models/auto.py#L164) applies
[QAT preparation](../../veomni/models/qat.py#L109) before FSDP2, including on meta.
Weight names stay unchanged; SwiGLU calls the fake-quant projections' forwards.
[The parallel builder](../../veomni/distributed/torch_parallelize.py#L1011) validates
topology and precision before sharding/loading. AdamW updates FP32 master shards.

For exact optimizer continuation, restore DCP after rebuilding the same recipe
and targets. Fake-quant scales are derived each forward, not persisted state.
Ordinary HF export retains recipe/target metadata in `config.json` and normally
rounds FP32 parameters to BF16, without optimizer state or packed MXFP4 weights.
Reloading through the model builder prepares the selected projections again.

## Validation

Run the [two-GPU acceptance](../../tests/models/test_mxfp4_qat_fsdp.py#L202) test:

```bash
CUDA_VISIBLE_DEVICES=0,1 pytest tests/models/test_mxfp4_qat_fsdp.py -q
```

The test covers NCCL on two CUDA ranks, generated Qwen3 modeling, two
accumulated microbatches, FP32 master updates, DCP/AdamW continuation and HF reload.
It declares a two-card CI plan, is selected by `tests/models/`, and skips without
suitable CUDA hardware.

Acceptance uses tiny random Qwen3 with eager ops and rebuilds DCP/AdamW state in
the same rank processes. It does not cover fresh-process or full trainer/data/
scheduler/RNG resume, convergence, speedup, peak memory or deployment compatibility.
