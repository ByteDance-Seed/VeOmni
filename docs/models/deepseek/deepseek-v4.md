# DeepSeek V4

## 1. Model introduction

DeepSeek V4 training with dedicated DSA, indexer, and mHC kernel selection.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [deepseek_v4.yaml](../../../configs/text/deepseek_v4.yaml) | conversation |
| [deepseek_v4_npu.yaml](../../../configs/text/deepseek_v4_npu.yaml) | plaintext |
| [deepseek_v4_indexer_loss.yaml](../../../configs/text/deepseek_v4_indexer_loss.yaml) | conversation |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id deepseek-ai/DeepSeek-V4-Flash-Base \
  --local_dir downloads
```

Prepare conversation data with the [Qwen3 walkthrough](../qwen/qwen3.md#download-dataset) or the small synthetic dataset in [Quick Start](../../get_started/quick_start.md).

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_text.py configs/text/deepseek_v4.yaml \
  --model.model_path downloads/DeepSeek-V4-Flash-Base \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/deepseek-v4 \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary YAML sets these values; review the linked configurations before
changing a checkpoint, sequence length, or parallel layout.

| Field | Checked-in value |
| --- | --- |
| `data.max_seq_len` | `2048` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `1` |
| `model.optimizer.type` | `muon` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |

GPU and NPU recipes select different kernels and input data. The GPU recipe uses eager top-level attention, fused Triton MoE, and TileLang DSA/mHC; the default Muon `gram_quack` backend has additional hardware/dependency requirements. Check [kernel selection](../../design/kernel_selection.md#deepseek-v4-dsa-and-mhc) before launching. The indexer-loss config is a separate experiment with a shortened checkpoint name, not the default full-model recipe.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
