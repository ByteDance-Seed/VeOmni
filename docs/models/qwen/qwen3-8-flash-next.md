# Qwen3.8 Flash Next

## 1. Model introduction

Training the Qwen4-Exp architecture with MoE expert parallelism and sharded PLE embeddings.

## 2. Variants and recipes

The primary checkpoint in this walkthrough is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Choose the configuration matching your task:

| Configuration | Input data |
| --- | --- |
| [qwen4exp.yaml](../../../configs/text/qwen4exp.yaml) | conversation |

These are configuration-backed recipes. They are not a certification of every
model size or hardware combination; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md).
Run from the repository root. Download the model with any access required by its publisher:

```bash
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3.8-Flash-Next \
  --local_dir downloads
```

Prepare conversation data with the [Qwen3 walkthrough](../qwen/qwen3.md#download-dataset) or the small synthetic dataset in [Quick Start](../../get_started/quick_start.md).

## 4. Launch training

Replace `/path/to/prepared-data` before running. Configure `NPROC_PER_NODE` for
your machine; for multiple nodes, also set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`
on each node. Choose a topology compatible with the SP/EP settings below.
The meta-initialized FSDP2 path requires more than one process.

```bash
bash train.sh tasks/train_text.py configs/text/qwen4exp.yaml \
  --model.model_path downloads/Qwen3.8-Flash-Next \
  --data.train_path /path/to/prepared-data \
  --train.checkpoint.output_dir outputs/qwen3-8-flash-next \
  --train.wandb.enable false
```

## 5. Recipe configuration

The primary recipe uses the settings below. Omitted parallel sizes default to
1 and an omitted optimizer type defaults to AdamW. Review the linked YAML before
changing a checkpoint, sequence length, or parallel layout.

| Field | Recipe setting |
| --- | --- |
| `data.max_seq_len` | `16384` |
| `model.accelerator.ulysses_size` | `1` |
| `model.accelerator.ep_size` | `4` |
| `model.optimizer.type` | `adamw` |
| `train.global_batch_size` | automatic (`null`) |
| `train.micro_batch_size` | `1` |

The checked-in recipe sets EP=4 and PLE=8 on separate ExtraParallel meshes, with `ep_sharded_stream_load: true`. Read the [PLE layout contract](../../design/qwen4_exp_ple_2d_parallelism.md) before choosing world size. Do not infer a minimum-memory setup from the parallel sizes; this is a large model. The recipe disables HF export and uses a ten-step run by default.

## 6. Validation and next steps

Start with a short run and inspect finite loss and gradient norms. When a save
is configured, verify the [checkpoint completion manifest](../../usage/checkpoint.md#completion)
before testing resume or exporting weights. Memory fit and model quality need
validation on your own hardware and dataset.

- [Parallelism and feature guides](../../key_features/index.md)
- [Checkpoint layout and resume](../../usage/checkpoint.md)
- [Add or extend a model](../../usage/support_new_models/guide_and_checklist.md)
