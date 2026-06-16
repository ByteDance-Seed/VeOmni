# Typical Usage: Qwen3-VL 8B Training on Ascend NPU

This document provides a complete step-by-step guide for training the Qwen3-VL 8B model on Ascend NPUs. Follow these instructions carefully to ensure a successful training experience.

## Prerequisites

- Ascend NPU environment with CANN 9.0.0 installed
- VeOmni framework installed (see [Installation](get_started_npu.md#installation) section)
- Sufficient storage space for dataset and model weights

## Step 1: Download Dataset

We'll use the COCO2017 dataset and ShareGPT4V annotations for training. Follow these steps to prepare the dataset:

```bash
# Download COCO2017 dataset
wget https://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download ShareGPT4V annotations
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
```

## Step 2: Preprocess Dataset

Modify the annotation file to match the expected format for VeOmni:

```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        # Update the image path to point to your downloaded COCO dataset
        new_item['images'] = [f'./train2017/{image_path.split("/")[-1]}']
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

## Step 3: Download Pre-trained Model

Download the Qwen3-VL-8B-Instruct model weights:

```bash
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir ./Qwen3-VL-8B-Instruct
```

## Step 4: Configure Training

VeOmni uses YAML configuration files for training. You can directly modify the configuration file at `configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml` to adjust parameters like batch size, learning rate, and other hyperparameters according to your needs.

### NPU-Friendly Operator Configurations

**Important**: The default configuration files are optimized for GPU environments. When running on Ascend NPUs, we recommend adding the following **minimal NPU-friendly operator configurations** to your model's YAML file. These settings optimize operator implementations specifically for NPU hardware and are generally applicable to most models in the repository:

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_2
    moe_implementation: fused_npu
    cross_entropy_loss_implementation: npu
    rms_norm_implementation: npu
    rotary_pos_emb_implementation: npu
    swiglu_mlp_implementation: eager           # no NPU backend available
    load_balancing_loss_implementation: eager  # triton-ascend not exposed as `triton`
    # Qwen3.5 GatedDeltaNet trio — only meaningful for Qwen3.5 MoE models;
    # pin to eager on NPU (no NPU kernel) and set train.dyn_bsz=False:
    rms_norm_gated_implementation: eager
    causal_conv1d_implementation: eager
    chunk_gated_delta_rule_implementation: eager
```

These configurations specify the optimal implementation for each operator type when running on NPUs:
- Use NPU-optimized implementations where available (rms_norm_implementation: npu)
- Fall back to compatible implementations for operations without NPU support (eager)
- Configure specialized settings for model-specific components (Qwen3.5 GatedDeltaNet)

**Note**: Some models with structurally-incompatible kernels (e.g., Wan rope_apply, Qwen2-VL multimodal RoPE) already include these NPU-friendly configurations in their default YAML files.

## Step 5: Start Training

Run the training command with the appropriate parameters:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft
```

## Step 6: Checkpoint Configuration

VeOmni automatically saves checkpoints during training. You can configure the checkpoint behavior in the YAML configuration file:

```yaml
train:
  checkpoint:
    output_dir: ./checkpoints
    save_steps: 1000
    save_epochs: 1
    save_hf_weights: true
```

Key checkpoint configuration parameters:
- `output_dir`: Directory to save checkpoints
- `save_steps`: Number of steps between checkpoint saves
- `save_epochs`: Number of epochs between checkpoint saves
- `save_hf_weights`: Whether to save Hugging Face model weights in addition to the VeOmni checkpoint format (only in the last checkpoint directory)
