(examples-qwen3)=

# Qwen3 training guide

## Download dataset
Download the fineweb dataset.

```shell
python3 scripts/download_hf_data.py \
  --repo_id HuggingFaceFW/fineweb \
  --local_dir ./fineweb/ \
  --allow_patterns sample/10BT/*
```
## Download Qwen3 model

### Qwen3-8B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-8B \
    --local_dir .
```

### Qwen3-30B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --local_dir .
```

Merge qwen3 moe model experts to support GroupGemm optimize.
``` shell
python3 scripts/moe_ckpt_merge/moe_merge.py --raw_hf_path Qwen3-30B-A3B-Instruct-2507  --merge_hf_path Qwen3-30B-A3B-Instruct-2507-merge
```

## Start training on GPU/NPU

### Qwen3-8B

```shell
bash train.sh tasks/train_torch.py configs/pretrain/qwen2_5.yaml \
    --model.model_path ./Qwen3-8B \
    --data.train_path ./fineweb/sample/10BT \
    --train.data_parallel_mode fsdp2 \
    --train.init_device meta \
    --train.use_wandb false
```

### Qwen3-30B

```shell
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_moe.yaml \
    --model.model_path ./Qwen3-30B-A3B-Instruct-2507-merge \
    --data.train_path ./fineweb/sample/10BT \
    --train.data_parallel_mode fsdp2 \
    --train.init_device meta \
    --train.use_wandb false
```
