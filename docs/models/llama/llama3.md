# Llama3

## 1. Model introduction

Continued pre-training on plaintext data using the Llama3 configuration.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [llama3.yaml](../../../configs/text/llama3.yaml) | plaintext |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id meta-llama/Meta-Llama-3-8B \
  --local_dir downloads
```

Prepare Parquet data with a `text` column using the [data guide](../../usage/data_packing_and_dyn_bsz.md).

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_text.py configs/text/llama3.yaml \
  --model.model_path downloads/Meta-Llama-3-8B \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/llama3 \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary recipe uses the settings below. Omitted parallel sizes default to
1 and an omitted optimizer type defaults to AdamW. Review the linked YAML before
changing a checkpoint, sequence length, or parallel layout.

| Field | Recipe setting |
| --- | --- |
| `data.max_seq_len` | `8192` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `1` |
| `model.optimizer.type` | `adamw` |
| `train.global_batch_size` | `512` |
| `train.micro_batch_size` | `1` |

Obtain access to the checkpoint before downloading. The recipe reads `text`, uses an 8192-token sequence length, and disables FSDP mixed precision in the checked-in YAML. Review precision and memory usage before scaling.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
