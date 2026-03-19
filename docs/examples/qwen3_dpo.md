# Qwen3 DPO training guide

DPO training with **Qwen3-0.6B** on the
[trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) dataset.

Config: [`configs/text/qwen3_dpo.yaml`](../../configs/text/qwen3_dpo.yaml)

---

## Step 1 — Prepare the dataset

```shell
python scripts/download_hf_data.py \
    --repo_id trl-lib/ultrafeedback_binarized \
    --local_dir ./ultrafeedback_binarized \
    --allow_patterns "*.parquet"
```

This downloads all train and test Parquet files from the Hub into `./ultrafeedback_binarized/`.

---

## Step 2 — Prepare the model

```shell
python scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-0.6B \
    --local_dir .
```

The script appends the model name to `--local_dir`, saving to `./Qwen3-0.6B`.

---

## Step 3 — Start DPO training

```shell
bash train.sh tasks/train_text_dpo.py configs/text/qwen3_dpo.yaml
```

Key config values (full DPO argument reference: [arguments.md — DPOConfig](../usage/arguments.md#dpo-arguments)):

| Parameter | Value |
|---|---|
| `model.model_path` | `./Qwen3-0.6B` |
| `data.train_path` | `./ultrafeedback_binarized` |
| `data.max_seq_len` | `2048` |
| `train.global_batch_size` | `8` |
| `train.micro_batch_size` | `1` |
| `train.num_train_epochs` | `1` |
| `dpo_config.beta` | `0.1` |
| `dpo_config.loss_type` | `sigmoid` |
| `train.checkpoint.output_dir` | `Qwen3-0.6B-dpo-ultrafeedback` |
| `train.wandb.project` | `VeOmni` |
| `train.wandb.name` | `Qwen3-0.6B-dpo-ultrafeedback` |

---

## Step 4 — Monitor training

Training outputs (DPO loss, chosen/rejected rewards, reward accuracy, grad norm) are printed every
step and logged to Weights & Biases.

---

## Checkpoints

Checkpoints are saved under `train.checkpoint.output_dir` every `save_steps` steps.
With `save_hf_weights: true`, a HuggingFace-compatible checkpoint is also written:

```
Qwen3-0.6B-dpo-ultrafeedback/
└── checkpoints/
    └── global_step_200/
        ├── ...          ← DCP distributed checkpoint
        └── hf_ckpt/     ← HuggingFace safetensors
```
