<span id="seed-oss-training-guide"></span>

# Seed-OSS

## 1. Model introduction

### Scope and prerequisites

Seed-OSS language-model training on FineWeb plain text (`data_type: plaintext`).

Configuration: [training YAML](../../../configs/text/seed_oss.yaml). Read the
[catalog prerequisites and validation scope](../index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## 2. Variants and recipes

| Variant / component | Model repository | Recipe scope |
| --- | --- | --- |
| Seed-OSS-36B-Instruct | [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) | Plain-text FineWeb training recipe |

This table identifies checkpoints used by the recipe. Full-size hardware validation
is separate from configuration availability; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md)
and run the commands from the repository root. Replace example filesystem paths
with your own model, dataset, and output locations.

### Download dataset

Download the [fineweb 10BT sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT) dataset.

### Download SeedOss model

```shell
python3 scripts/download_hf_model.py \
    --repo_id ByteDance-Seed/Seed-OSS-36B-Instruct \
    --local_dir .
```

## 4. Launch training

### Start training on GPU/NPU

```shell
bash train.sh tasks/train_text.py configs/text/seed_oss.yaml \
    --model.model_path ./Seed-OSS-36B-Instruct \
    --data.train_path ./fineweb/sample/10BT
```

## 5. Recipe configuration

| Configuration | Data | Sequence length | SP / EP | Global batch |
| --- | --- | --- | --- | --- |
| [seed_oss.yaml](../../../configs/text/seed_oss.yaml) | plaintext | 4096 | default / default | 512 |

These are checked-in defaults, not minimum hardware requirements. Match the
world size, batch size, and parallel topology to your checkpoint and memory budget.

## 6. Validation and next steps

### Check outputs and continue

Use the [run checks](../index.md#choose-hardware-and-check-a-run) and
[checkpoint completion contract](../../usage/checkpoint.md#completion) for training.

### Related guides

- [Checkpoint save, resume, and export](../../usage/checkpoint.md)
- [LoRA](../../key_features/lora.md)
- [Model families](../index.md)
