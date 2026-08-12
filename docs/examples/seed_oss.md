# Seed-OSS training guide

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
