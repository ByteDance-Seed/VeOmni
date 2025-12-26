# Qwen3-VL MoE Training Guide

## 1. Download Qwen3-VL MoE Model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --local_dir .
```

## 2. Merge Qwen3-VL MoE Model Experts (Optional)

If you want to use GroupGemm optimization for MoE experts, merge the model:

```shell
python3 scripts/moe_ckpt_merge/moe_merge.py \
  --raw_hf_path Qwen3-VL-30B-A3B-Instruct \
  --merge_hf_path Qwen3-VL-30B-A3B-Instruct-merge
```

Then update the `model_path` in your config to use the merged model.

## 3. Prepare Dataset

Download the [ShareGPT4V-small](https://github.com/iqiancheng/ShareGPT4V-small) dataset.

The dataset supports relative image paths relative to the `train_path` directory. For example, if your `train_path` is `/path/to/ShareGPT4V-small-coco-128.jsonl` and an image path in the dataset is `coco/train2017/image.jpg`, it will be automatically resolved to `/path/to/coco/train2017/image.jpg`.

## 4. Configure Training

Update the config file `configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml`:

```yaml
model:
  model_path: Qwen3-VL-30B-A3B-Instruct  # or Qwen3-VL-30B-A3B-Instruct-merge if merged
  moe_implementation: fused
  attn_implementation: flash_attention_2

data:
  train_path: /path/to/ShareGPT4V-small-coco-128.jsonl
  data_type: conversation
  source_name: sharegpt4v_sft
  chat_template: qwen2_5vl
  max_seq_len: 4096
  dataloader_type: native
  datasets_type: iterable 

train:
  output_dir: qwen3_vl_moe_sft
  data_parallel_mode: fsdp2
  enable_reentrant: false
  use_wandb: true
  wandb_project: qwen3_vl_moe
  wandb_name: qwen3_vl_moe
  rmpad: false
  rmpad_with_pos_ids: true
  expert_parallel_size: 1
  freeze_vit: false
  lr: 1.0e-5
  lr_decay_style: cosine
  num_train_epochs: 2
  micro_batch_size: 1
  global_batch_size: 16
  max_steps: 500
  init_device: meta
  ckpt_manager: dcp
  save_hf_weights: false
```

## 5. Train Qwen3-VL MoE Model

```shell
bash train.sh tasks/omni/train_qwen_vl.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml
```
