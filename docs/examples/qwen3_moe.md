# Qwen3 MoE training guide

## Scope and prerequisites

Train a Qwen3 MoE language model with the existing
[SFT configuration](../../configs/text/qwen3-moe.yaml). Install the
[environment for your hardware](../hardware_support/index.md) first and prepare
conversation data using the [Qwen3 data instructions](qwen3.md#download-dataset).
For a small first run, use the [Quick Start](../get_started/quick_start.md).

Choose enough accelerator memory for the actual model size, optimizer state,
sequence length, and parallel topology. This recipe does not establish a minimum
GPU count for the 30B or 235B checkpoints.

## Download the model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-30B-A3B \
  --local_dir .
```

The helper creates `./Qwen3-30B-A3B`. VeOmni's runtime checkpoint converter
loads the stock HF expert weights into the fused expert layout. An offline
merge is not required. See [MoE weight loading](../transformers_v5/transformers_v5_moe_weight_loading.md)
for supported layouts and export conversion.

## Launch training

After creating `tulu-first2000.parquet` as described in the data instructions:

```shell
bash train.sh tasks/train_text.py configs/text/qwen3-moe.yaml \
  --model.model_path ./Qwen3-30B-A3B \
  --data.train_path ./tulu-first2000.parquet \
  --train.checkpoint.output_dir outputs/qwen3-moe
```

The explicit model path overrides the historical `-merge` directory name in
this configuration. Inspect the YAML before launch and adjust batch size and
parallelism for your topology. To enable or tune EP, follow
[EP with FSDP2](../key_features/ep_fsdp2.md) and the
[ExtraParallel guide](../key_features/extra_parallel.md).

## Check outputs and continue

Inspect `log.txt` for finite loss and gradient norms, then verify the configured
saves under `outputs/qwen3-moe/checkpoints/`. The
[checkpoint guide](../usage/checkpoint.md) explains completion metadata, resume
state, and HF exports. A successful short run checks the training path; validate
convergence on your own dataset before scaling up.

- [MoE LoRA](../key_features/lora.md#5-moe-lora) covers adapter training.
- [Fused MoE implementation](../design/fused_moe_kernels.md) explains the kernels.
- [Model catalog](index.md) lists related Qwen3 recipes.
