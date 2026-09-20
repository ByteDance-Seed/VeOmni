# Qwen-Image

## 1. Model introduction

Diffusion transformer SFT and LoRA for Qwen-Image.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [qwen_image_sft.yaml](../../../configs/dit/qwen_image_sft.yaml) | diffusion |
| [qwen_image_lora.yaml](../../../configs/dit/qwen_image_lora.yaml) | diffusion |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen-Image \
  --local_dir downloads
```

Prepare image/prompt data using the [Qwen-Image LoRA walkthrough](../../key_features/lora.md#64-qwen-image-lora-dit-fsdp2). Use the recipe’s condition model and preprocessing settings.

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_dit.py configs/dit/qwen_image_sft.yaml \
  --model.model_path downloads/Qwen-Image/transformer \
  --model.condition_model_path downloads/Qwen-Image \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/qwen-image \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary recipe uses the settings below. Omitted parallel sizes default to
1 and an omitted optimizer type defaults to AdamW. Review the linked YAML before
changing a checkpoint, sequence length, or parallel layout.

| Field | Recipe setting |
| --- | --- |
| `data.max_seq_len` | `4096` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `1` |
| `model.optimizer.type` | `adamw` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |

The transformer and condition model have different paths. The YAML also names a model config and image resolution; keep them aligned with the checkpoint. Follow the [Qwen-Image LoRA walkthrough](../../key_features/lora.md#64-qwen-image-lora-dit-fsdp2) for the image/prompt training data and adapter settings.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
