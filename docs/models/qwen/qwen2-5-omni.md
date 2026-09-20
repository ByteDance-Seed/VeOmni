# Qwen2.5 Omni

## 1. Model introduction

Multimodal conversation training with image, video, and audio inputs.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [qwen25_omni.yaml](../../../configs/multimodal/qwen25_omni/qwen25_omni.yaml) | conversation |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen2.5-Omni-7B \
  --local_dir downloads
```

Prepare conversations and media paths with the [multimodal data guide](../../usage/multimodal_data_processing.md). For image/video examples, see [Qwen3 VL data preparation](../qwen/qwen3-vl.md#download-dataset). Audio inputs additionally need the family’s audio processor settings.

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen25_omni/qwen25_omni.yaml \
  --model.model_path downloads/Qwen2.5-Omni-7B \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/qwen2-5-omni \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary recipe uses the settings below. Omitted parallel sizes default to
1 and an omitted optimizer type defaults to AdamW. Review the linked YAML before
changing a checkpoint, sequence length, or parallel layout.

| Field | Recipe setting |
| --- | --- |
| `data.max_seq_len` | `8196` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `1` |
| `model.optimizer.type` | `adamw` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |

The data config mixes text, image, video, and voice-assistant sources. Update all source paths before launch. Preserve the sample rate, frame controls, and `use_audio_in_video` setting expected by the processor.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
