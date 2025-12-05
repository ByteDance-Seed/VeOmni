# Qwen3 VL training guide

## Download dataset

1. Download the ShareGPT4V-small-coco-128.jsonl:  [ShareGPT4V-small-coco-128.jsonl](https://github.com/iqiancheng/ShareGPT4V-small/blob/main/ShareGPT4V-small-coco-128.jsonl)

2. Download the COCO 2017 training images:  [COCO train2017 dataset](https://images.cocodataset.org/zips/train2017.zip)

3. Final directory structure should be like this:
> ```
> VeOmni
> ├—— ShareGPT4V-small-coco-128.jsonl
> └—— coco/
>     └—— train2017/
>         ├—— 000000000009.jpg
>         ├—— 000000000026.jpg
>         └—— ... (more images)
> ```

## Download qwen3vl model
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir .
```

## Start training on NPU

```shell
bash train.sh tasks/omni/train_qwen_vl.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./ShareGPT4V-small-coco-128.jsonl \
    --data.dataloader_type native \
    --data.dataset_type iterable \
    --data.sourcename sharegpt4v_sft \
    --train.use_wandb false
```