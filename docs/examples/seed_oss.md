# Seed-OSS training guide

## Scope and prerequisites

Text SFT for Seed-OSS with conversation data.

Configuration: [training YAML](../../configs/text/seed_oss.yaml). Read the
[catalog prerequisites and validation scope](index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## Download dataset

Download the [fineweb 10BT sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT) dataset.

## Download SeedOss model

```shell
python3 scripts/download_hf_model.py \
    --repo_id ByteDance-Seed/Seed-OSS-36B-Instruct \
    --local_dir .
```

## Start training on GPU/NPU

```shell
bash train.sh tasks/train_text.py configs/text/seed_oss.yaml \
    --model.model_path ./Seed-OSS-36B-Instruct \
    --data.train_path ./fineweb/sample/10BT
```

## Check outputs and continue

Use the [run checks](index.md#choose-hardware-and-check-a-run) and
[checkpoint completion contract](../usage/checkpoint.md#completion) for training.
For preprocessing or inference, inspect the stage-specific outputs described above.
