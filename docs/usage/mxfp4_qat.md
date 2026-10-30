# Dense Qwen3 MXFP4 QAT

`torchao_mxfp4` applies quantization-aware training to selected Dense Qwen3
projections. It simulates MXFP4 activation and weight rounding during forward
and uses a straight-through gradient estimator. Parameters remain ordinary
floating-point tensors, so the existing optimizer and checkpoint paths apply.

This is the first implementation slice of the
[low-precision RFC](https://github.com/ByteDance-Seed/VeOmni/issues/1360).
The recipe uses torchao's **emulated** MXFP4 implementation, not native FP4
training GEMMs. It makes no training speedup, convergence or downstream
rollout-compatibility claim.

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
| Checkpoints | Existing DCP for exact optimizer continuation; ordinary HF floating-point export |

The emulated recipe does not require Blackwell's native FP4 instructions.
Use CUDA devices with BF16 support for training. CPU emulation does not verify
CUDA kernels, NCCL collectives or physical multi-GPU FSDP2.

The first implementation rejects other model families, MoE, shared selected
weights, projections with biases, and inputs whose projection width is not
divisible by 32. `lm_head` is excluded because the training loss can read its
weight directly instead of calling the projection's forward.

Distributed use rejects DDP, HSDP, SP/CP, TP/PP, ExtraParallel, LoRA, FSDP
parameter CPU offload and `torch.compile`. The example keeps activation
offload disabled; combining activation offload with this recipe has not been
validated. Native DeepSeek `fp8_blockwise` QAT remains a separate recipe.

## Configure training

Install the normal GPU environment:

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

Patterns match complete module names. Every pattern must select at least one
supported projection. To include attention, add
`model.layers.*.self_attn.q_proj`, `k_proj`, `v_proj` and `o_proj` as separate
patterns with the same prefix. The default recipe remains `none`.

Start from [the Qwen3 example](../../configs/text/qwen3_mxfp4_qat.yaml), set
`model.model_path` and `data.train_path` to the desired checkpoint and prepared
conversation dataset, then run on a host with at least two CUDA devices:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3_mxfp4_qat.yaml
```

The example's learning rate and training duration are starting settings, not a
validated convergence recipe.

## Construction and checkpoints

1. [The foundation-model builder](../../veomni/models/auto.py#L164) loads the model
   and applies [the QAT preparation helper](../../veomni/models/qat.py#L119) before
   returning it, including when parameters are still on the meta device.
2. Selected projections become torchao fake-quant modules without changing
   their weight names. SwiGLU calls their forwards instead of bypassing them
   through the plain-Linear fused path.
3. [The parallel builder](../../veomni/distributed/torch_parallelize.py#L1011)
   validates the supported topology and precision, then applies normal FSDP2
   sharding and weight loading. AdamW updates the FP32 master shards.
4. DCP saves model and optimizer state using the existing checkpointer.
   Rebuild with the same recipe and target config before restoring a training
   checkpoint. Fake-quant scales are derived from current operands each
   forward; they are not separate persistent training state.

Ordinary HF export keeps the recipe and target metadata in `config.json`.
Loading through VeOmni's foundation-model builder prepares the same selected
projections again. The existing distributed HF writer normally rounds FP32
parameters to BF16; it does not save packed MXFP4 weights or optimizer state.
Use DCP when exact training continuation is required.

## Offline MXFP4 artifact

After saving an ordinary HF checkpoint, create a separate packed torchao
artifact:

```bash
PYTHONPATH=. python scripts/export_mxfp4.py \
  --input /path/to/hf_checkpoint \
  --output /path/to/empty_mxfp4_directory
```

The converter operates on an unsharded model. It copies the model, reproduces
BF16-facing parameter rounding, and converts only the selected projections.
The training checkpoint is not modified. Allow host memory for both the
loaded model and conversion copy.

The output contains `model.pt`, `config.json` and `mxfp4.json`. It is a
versioned torchao inference artifact with packed MXTensor weights, **not** a
standard Hugging Face, vLLM or SGLang quantized checkpoint. Reload it using:

```python
from veomni.models.qat import load_mxfp4

model = load_mxfp4("/path/to/mxfp4_directory", device="cpu")
```

`load_mxfp4` returns an evaluation model with gradients disabled and uses
torchao's emulated inference path. This format does not establish train/rollout
log-probability agreement or support repeated rollout refits. Keep its pinned
torchao version when saving and loading artifacts.

## Validation

CPU operator, model and offline conversion cases live in
[`test_mxfp4_qat.py`](../../tests/models/test_mxfp4_qat.py#L15). The physical
two-GPU acceptance case is
[`test_mxfp4_qat_fsdp.py`](../../tests/models/test_mxfp4_qat_fsdp.py#L182):

```bash
pytest tests/models/test_mxfp4_qat.py -q
CUDA_VISIBLE_DEVICES=0,1 pytest tests/models/test_mxfp4_qat_fsdp.py -q
```

The second test uses two distinct CUDA ranks with NCCL, generated Qwen3
modeling, two accumulated microbatches, FP32 master updates, exact DCP/AdamW
continuation and ordinary HF export/reload. It declares a two-card CI launch
plan and skips on CPU/NPU or insufficient hardware. Both files are covered by
the existing `tests/models/` directory selection.

The multi-GPU test is prepared but has not yet been executed on physical
GPUs for this change. Collection or a CPU skip is not a passing distributed
acceptance result. Representative convergence, throughput, peak memory and
deployment-format compatibility remain outside the current evidence.
