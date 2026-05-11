# Get Started with Ascend NPU

## Overview

This guide provides comprehensive information for using VeOmni framework with Ascend NPUs. Ascend NPUs are high-performance AI accelerators designed for efficient model training and inference. VeOmni's support for Ascend NPUs enables users to leverage these powerful accelerators for distributed training of multi-modal models.

### What This Guide Covers

- **Installation**: Step-by-step instructions for setting up VeOmni on Ascend NPU platforms
- **Supported Models**: List of multi-modal models that can be trained on Ascend NPUs
- **Environment Configuration**: Important environment variables and settings for optimal performance
- **Typical Usage**: Complete example for training a Qwen3-VL 8B model on Ascend NPUs
- **FAQ**: Common questions and solutions for Ascend NPU usage

## Key Updates

2026/5/11: Veomni provides images of the version of Ascend Cann9.0.0.
2025/12/23: VeOmni supports training on Ascend NPU.

## Installation

VeOmni supports two installation methods for Ascend NPUs: `uv` (recommended for faster installation) and `pip`. Note that ARM architecture machines only support `uv` installation.

### Installation Options

- **x86 Architecture**: Supports both `uv` and `pip` installation methods
- **ARM Architecture**: Only supports `uv` installation method

### Detailed Installation Guide

Please refer to the specific installation guides based on your architecture:

- [Installation with Ascend NPU (x86)](../get_started/installation/install_ascend_x86.md)
- [Installation with Ascend NPU (ARM)](../get_started/installation/install_ascend_arm.md)

### Docker Support

VeOmni also provides Docker support for Ascend NPUs. For detailed instructions on building and using Ascend Docker images, please refer to:

- [Ascend A3 Docker Image Build and Usage Guide](../get_started/AscendDockerUse/build_a3_docker.md)
- [Ascend A2 Docker Image Build and Usage Guide](../get_started/AscendDockerUse/build_a2_docker.md)

## Supported Models

VeOmni supports a wide range of models on Ascend NPUs, including large language models, multimodal models, and diffusion models. Below is a comprehensive list of supported models with their features:

| Model                | Model Size       | Support | FSDP1 | FSDP2 | EP | SP | Note                                           |
|----------------------|------------------|---------|-------|-------|----|----|------------------------------------------------|
| [Qwen3](../examples/qwen3.md) | 8B               | ✅       |       | ✅     |    |    |
|                      | 30B              | ✅       |       | ✅     |    |    |                                                |
| [Qwen3.5](../examples/qwen3.md) | 9B    | ✅        |         |       | ✅     |    |    |
|                      | 35B              | ✅       |       | ✅     |    |    |                                                |
| [Qwen3-MoE](../examples/qwen3.md) | 30B-A3B          | ✅       |       | ✅     | ✅  |    |
|                      | 235B-A22B        | ✅       |       | ✅     | ✅  |    |                                                |
| [Qwen3-VL](../examples/qwen3_vl.md) | 8B               | ✅       |       | ✅     |    | ✅  |                               |
|                      | 30B              | ✅       |       | ✅     | ✅  | ✅  |                                                |
| Qwen3-VL-MoE         | 30B-A3B          | ✅       |       | ✅     | ✅  | ✅  |                               |
|                      | 235B-A22B        | ✅       |       | ✅     | ✅  | ✅  |                                                |
| Qwen3-Omni-MoE       | 30B-A3B          | ✅       |       | ✅     | ✅  | ✅  |                               |
|                      | 235B-A22B        | ✅       |       | ✅     | ✅  | ✅  |                               |
| [Qwen3-DPO](../examples/qwen3_dpo.md) | 0.6B             | ✅       |       | ✅     |    |    |                               |
| [Wan2.1](../examples/wan2.1.md)    | 14B              | ✅       | ✅     |       |    | ✅  |                               |

**Legend:**
- **FSDP1**: Fully Sharded Data Parallel version 1
- **FSDP2**: Fully Sharded Data Parallel version 2 (recommended)
- **EP**: Expert Parallel - for MoE models
- **SP**: Sequence Parallel - enables longer sequence training

For detailed configuration files and training examples, please refer to the [configs](https://github.com/ByteDance-Seed/VeOmni/tree/main/configs) directory in the repository.

## Ascend Environment Variables

### CPU_AFFINITY_CONF

```shell
export CPU_AFFINITY_CONF=1
```
Enable coarse-grained or fine-grained CPU core binding. This configuration helps prevent thread contention, improves cache hit rates, avoids memory access across different NUMA (Non-Uniform Memory Access) nodes, and reduces task scheduling overhead—collectively optimizing task execution efficiency.
Parameter Settings:

* `0`: Disable the binding function. Default is `0`.
* `1`: Enable coarse-grained kernel binding.
* `2`: Enable fine-grained kernel binding.

### PYTORCH_NPU_ALLOC_CONF

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

`expandable_segments:<value>`: Enable the memory pool extension segment feature.  
* `True`: This configuration instructs the cache allocator to create specific memory blocks with the capability to be extended later. This allows for more efficient handling of scenarios where the required memory size frequently changes during runtime.  
* `False`: The memory pool extension segment feature is disabled, and the original memory allocation method is used. Default is `False`.

### MULTI_STREAM_MEMORY_REUSE

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

Refer to: https://github.com/ByteDance-Seed/VeOmni/issues/575

## Typical Usage: Qwen3-VL 8B Training on Ascend NPU

This section provides a complete step-by-step guide for training the Qwen3-VL 8B model on Ascend NPUs. Follow these instructions carefully to ensure a successful training experience.

### Prerequisites

- Ascend NPU environment with CANN 8.3.RC1 installed
- VeOmni framework installed (see [Installation](#installation) section)
- Sufficient storage space for dataset and model weights

### Step 1: Download Dataset

We'll use the COCO2017 dataset and ShareGPT4V annotations for training. Follow these steps to prepare the dataset:

```bash
# Download COCO2017 dataset
wget https://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download ShareGPT4V annotations
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
```

### Step 2: Preprocess Dataset

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

### Step 3: Download Pre-trained Model

Download the Qwen3-VL-8B-Instruct model weights:

```bash
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir ./Qwen3-VL-8B-Instruct
```

### Step 4: Configure Training

VeOmni uses YAML configuration files for training. You can directly modify the configuration file at `configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml` to adjust parameters like batch size, learning rate, and other hyperparameters according to your needs.

### Step 5: Start Training

Run the training command with the appropriate parameters:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
```

### Step 6: Checkpoint Configuration

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
- `save_hf_weights`: Whether to save Hugging Face model weights in addition to the VeOmni checkpoint format(only in the last checkpoint directory)

## Common Precision Issues and Solutions

For detailed guidance on how to identify and resolve precision issues on Ascend NPUs, including version compatibility checks, debugging tools, and common issue patterns, please refer to our dedicated documentation:

[Precision Analysis and Troubleshooting Guide](precision_analysis.md)

## Ascend Profiling Collection and Analysis

For detailed guidance on how to collect and analyze profiling data on Ascend NPUs, including configuration settings, key metrics, and performance optimization strategies, please refer to our dedicated documentation:

[Profiling Collection, Analysis and Optimization Guide](profiling_analysis.md)

## FAQ

For answers to frequently asked questions about using VeOmni with Ascend NPUs, including memory management, multi-node training configuration, operator selection, and more, please refer to our dedicated FAQ document:

[FAQ: Common Issues and Solutions for Ascend NPU](FAQ.md)

## Declarations

The Ascend support code, Dockerfile and image provided in the documentation are for reference only. If you intend to use them in a production environment, please contact the official channels. Thank you.
