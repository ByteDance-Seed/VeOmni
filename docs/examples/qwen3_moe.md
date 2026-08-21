# Qwen3 MoE training guide

This guide trains Qwen3-30B-A3B directly from its Hugging Face checkpoint.
VeOmni converts the checkpoint's per-expert tensors to its fused expert layout
at load time, so offline checkpoint merging is not required.

## Prepare the model and dataset

Download the model:

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-30B-A3B \
    --local_dir .
```

The model is saved to `./Qwen3-30B-A3B`, matching the default
`model.model_path` in `configs/text/qwen3-moe.yaml`.

Download the training dataset:

```shell
python3 scripts/download_hf_data.py \
    --repo_id allenai/tulu-3-sft-mixture \
    --local_dir ./tulu-3-sft-mixture
```

## Start training

The example config selects FSDP2, FlashAttention 2, and the fused Triton MoE
backend:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml
```

Override `model.model_path` or `data.train_path` on the command line when the
download locations differ:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml \
    --model.model_path /path/to/Qwen3-30B-A3B \
    --data.train_path /path/to/tulu-3-sft-mixture
```

On Ascend NPU, set `model.ops_implementation.moe_implementation` to
`fused_npu`. See [Kernel Selection in VeOmni](../design/kernel_selection.md)
for the complete backend matrix.

## Expert layout and fused dispatch

VeOmni supports `transformers==5.9.0`. In that version,
[`Qwen3MoeExperts`](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L214-L257)
already stores all experts in two three-dimensional parameters:

```text
gate_up_proj  [num_experts, 2 * intermediate_size, hidden_size]
down_proj     [num_experts, hidden_size, intermediate_size]
```

The upstream eager forward still loops over the experts that receive tokens.
VeOmni's Patchgen-generated model preserves the same parameter layout and adds
an `OpSlot("moe_experts", "standard")` guard. When a fused backend is selected,
the guard dispatches to the configured GPU or NPU fused MoE implementation;
with `moe_implementation: eager`, the reference PyTorch loop remains available
for correctness checks.

The integration is defined by:

- `veomni/models/transformers/qwen3_moe/qwen3_moe_gpu_patch_gen_config.py`
- `veomni/models/transformers/qwen3_moe/qwen3_moe_npu_patch_gen_config.py`
- `veomni/ops/kernels/moe/__init__.py`

Do not edit the generated modeling files directly.

## Checkpoint conversion

Published Qwen3-MoE checkpoints store separate tensors for each expert:

```text
experts.{j}.gate_proj.weight
experts.{j}.up_proj.weight
experts.{j}.down_proj.weight
```

At load time, `Qwen3MoeCheckpointTensorConverter` stacks the expert tensors and
merges the gate and up projections into `gate_up_proj`. The stock Hugging Face
checkpoint can therefore be passed directly to training.

`scripts/moe_ckpt_merge/moe_merge.py` is deprecated. It may still be used as a
one-time optimization when the same very large checkpoint is loaded repeatedly,
but it is not a prerequisite.

See [Transformers v5 MoE Weight Loading](../transformers_v5/transformers_v5_moe_weight_loading.md)
for the full load, save, and reverse-conversion matrix.
