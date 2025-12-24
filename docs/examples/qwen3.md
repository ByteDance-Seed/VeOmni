(examples-qwen3)=

# Qwen3 training guide

## Download dataset
Download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

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
bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml \
    --model.model_path ./Qwen3-8B \
    --data.train_path ./tulu-3-sft-mixture/data \
    --train.data_parallel_mode fsdp2 \
    --train.init_device meta \
    --train.use_wandb false
```

### Qwen3-30B

```shell
bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml \
    --model.model_path ./Qwen3-30B-A3B-Instruct-2507-merge \
    --model.moe_implementation fused \
    --data.train_path ./tulu-3-sft-mixture/data \
    --train.data_parallel_mode fsdp2 \
    --train.init_device meta \
    --train.global_batch_size 16 \
    --train.use_wandb false
```
