# Dense Qwen3 MXFP4 QAT

`torchao_mxfp4` simulates MXFP4 activation and weight rounding in selected Dense
Qwen3 projections, using straight-through gradients and ordinary floating-point
parameters. This initial [low-precision RFC](https://github.com/ByteDance-Seed/VeOmni/issues/1360)
recipe uses torchao's **emulated** MXFP4, without native FP4 training GEMMs or a
Blackwell requirement. Training requires NVIDIA CUDA devices with BF16 support.

## Recipe and scope

| Property | Initial recipe |
| --- | --- |
| Model | Standalone Dense Qwen3 |
| Targets | Explicit decoder attention or MLP projection module patterns |
| Quantization | MXFP4 E2M1 activations and weights, groups of 32, RCEIL scale calculation |
| Dependency | `torchao==0.18.0`, installed by the GPU extra |
| Training | NVIDIA CUDA, FSDP2, BF16 unsharded parameters, FP32 masters and gradient reduction |
| CPU | Eager numerical checks and offline artifact conversion/reload |
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

The example's learning rate and duration are starting settings, not a validated convergence recipe.

## Construction and checkpoints

[The model builder](../../veomni/models/auto.py#L164) applies
[QAT preparation](../../veomni/models/qat.py#L119) before FSDP2, including on meta.
Weight names stay unchanged; SwiGLU calls the fake-quant projections' forwards.
[The parallel builder](../../veomni/distributed/torch_parallelize.py#L1011) validates
topology and precision before sharding/loading. AdamW updates FP32 master shards.

For exact optimizer continuation, restore DCP after rebuilding the same recipe
and targets. Fake-quant scales are derived each forward, not persisted state.
Ordinary HF export retains recipe/target metadata in `config.json` and normally
rounds FP32 parameters to BF16, without optimizer state or packed MXFP4 weights.
Reloading through the model builder prepares the selected projections again.

## Offline MXFP4 artifact

Convert an ordinary HF checkpoint into a separate CPU packed torchao artifact:

```bash
PYTHONPATH=. python scripts/export_mxfp4.py \
  --input /path/to/hf_checkpoint \
  --output /path/to/empty_mxfp4_directory
```

Conversion copies an unsharded model, reproduces BF16-facing rounding and packs
only selected projections, leaving the training checkpoint unchanged. Allow host
memory for the loaded model and copy. Output: `model.pt`, `config.json`, `mxfp4.json`.

This versioned MXTensor artifact is torchao-specific, **not** a standard Hugging
Face, vLLM or SGLang quantized checkpoint. Reload with the same pinned torchao version:

```python
from veomni.models.qat import load_mxfp4

model = load_mxfp4("/path/to/mxfp4_directory", device="cpu")
```

The loaded model uses emulated inference with gradients disabled. This format
does not establish train/rollout log-probability agreement or support repeated rollout refits.

## Validation

Run the [CPU roundtrip](../../tests/models/test_mxfp4_qat.py#L48) and
[two-GPU acceptance](../../tests/models/test_mxfp4_qat_fsdp.py#L183) tests:

```bash
pytest tests/models/test_mxfp4_qat.py -q
CUDA_VISIBLE_DEVICES=0,1 pytest tests/models/test_mxfp4_qat_fsdp.py -q
```

The second test covers NCCL on two CUDA ranks, generated Qwen3 modeling, two
accumulated microbatches, FP32 master updates, DCP/AdamW continuation and HF reload.
It declares a two-card CI plan and skips without suitable CUDA hardware; both tests
are selected by `tests/models/`. CPU checks do not validate CUDA or collectives.

Acceptance uses tiny random Qwen3 with eager ops and rebuilds DCP/AdamW state in
the same rank processes. It does not cover fresh-process or full trainer/data/
scheduler/RNG resume, convergence, speedup, peak memory or deployment compatibility.
