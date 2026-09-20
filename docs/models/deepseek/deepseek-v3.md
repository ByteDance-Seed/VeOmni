# DeepSeek V2 / V3 / R1

## 1. Model introduction

DeepSeek language-model training and V3 LoRA with fused expert weights.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [deepseek.yaml](../../../configs/text/deepseek.yaml) | plaintext |
| [deepseek_v3_lora.yaml](../../../configs/text/deepseek_v3_lora.yaml) | conversation |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id deepseek-ai/DeepSeek-V3 \
  --local_dir downloads
```

Prepare Parquet data with a `text` column using the [data guide](../../usage/data_packing_and_dyn_bsz.md).

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_text.py configs/text/deepseek.yaml \
  --model.model_path downloads/DeepSeek-V3 \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/deepseek-v3 \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary YAML sets these values; review the linked configurations before
changing a checkpoint, sequence length, or parallel layout.

| Field | Checked-in value |
| --- | --- |
| `data.max_seq_len` | `8192` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `8` |
| `model.optimizer.type` | `adamw` |
| `train.global_batch_size` | `32` |
| `train.micro_batch_size` | `1` |

The base recipe has no model path set: provide the checkpoint explicitly. It uses plain text and EP=8. The V3 LoRA recipe instead uses conversations. Stock HF expert weights are converted at load time; follow the [MoE weight-loading contract](../../transformers_v5/transformers_v5_moe_weight_loading.md). Use the separate V4 recipe for V4-specific attention and kernels.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
