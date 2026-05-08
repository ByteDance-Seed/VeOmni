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

## Supported Models

VeOmni supports a wide range of models on Ascend NPUs, including large language models, multimodal models, and diffusion models. Below is a comprehensive list of supported models with their features:

| Model                               | Model Size                    | Support | FSDP1 | FSDP2 | EP | SP | Note |
|-------------------------------------|-------------------------------|---------|-------|-------|----|----|------|
| [DeepSeek2.5/3/R1](https://huggingface.co/deepseek-ai) | 236B/671B                     | ✅       | ✅     | ✅     |    |    |      |
| [Llama3-3.3](https://huggingface.co/meta-llama) | 1B/3B/8B/70B                  | ✅       | ✅     | ✅     |    |    |      |
| [Qwen2-3](https://huggingface.co/Qwen) | 0.5B/1.5B/3B/7B/14B/32B/72B   | ✅       |       | ✅     |    |    |      |
| [Qwen3](../examples/qwen3.md)       | 8B                            | ✅       |       | ✅     |    |    |      |
|                                     | 30B                           | ✅       |       | ✅     |    |    |      |
| [Qwen2-3 VL/QVQ](https://huggingface.co/Qwen) | 2B/3B/7B/32B/72B              | ✅       |       | ✅     |    | ✅  |      |
| [Qwen3 VL](../examples/qwen3_vl.md) | 8B                            | ✅       |       | ✅     |    | ✅  |      |
|                                     | 30B                           | ✅       |       | ✅     | ✅  | ✅  | MoE model |
| [Qwen3-MoE](https://huggingface.co/Qwen) | 30BA3B/235BA22B               | ✅       |       | ✅     | ✅  |    |      |
| [Qwen2-3 Omni](https://huggingface.co/Qwen) | 7B/30BA3B                     | ✅       |       | ✅     | ✅  | ✅  | Multimodal |
| [Wan2.1](../examples/wan2.1.md)     | 14B                           | ✅       | ✅     |       |    | ✅  | I2V model |

**Legend:**
- **FSDP1**: Fully Sharded Data Parallel version 1
- **FSDP2**: Fully Sharded Data Parallel version 2 (recommended)
- **EP**: Expert Parallel (for MoE models)
- **SP**: Sequence Parallel

For detailed configuration files and training examples, please refer to the [configs](https://github.com/ByteDance-Seed/VeOmni/tree/main/configs) directory in the repository.

## Environment Variables

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
    --data.dataloader.num_workers 8 \
    --train.micro_batch_size 3
```

### Step 6: Checkpoint Configuration

VeOmni automatically saves checkpoints during training. You can configure the checkpoint behavior in the YAML configuration file:

```yaml
train:
  checkpoint:
    output_dir: ./checkpoints
    save_strategy: steps
    save_steps: 1000
    keep_checkpoint_max: 5
    save_on_train_end: true
```

Key checkpoint configuration parameters:
- `output_dir`: Directory to save checkpoints
- `save_strategy`: When to save checkpoints ("steps" or "epoch")
- `save_steps`: Number of steps between checkpoint saves (if save_strategy is "steps")
- `keep_checkpoint_max`: Maximum number of checkpoints to keep
- `save_on_train_end`: Whether to save a checkpoint at the end of training

## Common Precision Issues and Solutions

When training models on Ascend NPUs, you may encounter precision mismatches compared to baseline results (e.g., on GPUs). This section provides guidance on how to identify and resolve common precision issues.

### Version Compatibility Check

Precision issues often arise due to version mismatches between software components. Follow these steps to verify compatibility:

1. **Check CANN Version**: Ensure you're using a compatible CANN version. VeOmni recommends CANN 8.3.RC1 for optimal performance and compatibility.

   ```bash
   # Check CANN version
   cat /usr/local/Ascend/cann/version.txt
   ```

2. **Check PTA Version**: The Precision Tuning Assistant (PTA) version should match your CANN version.

   ```bash
   # Check PTA version
   pip show ascend-pta
   ```

3. **Rollback Test**: If you recently upgraded your CANN or PTA version, try rolling back to a previous stable version to determine if the issue was introduced in the new version.

### Precision Debugging Tools

Ascend provides several tools to help diagnose precision issues:

#### 1. Precision Tuning Assistant (PTA)

PTA is a comprehensive tool for analyzing and resolving precision issues on Ascend NPUs. It can help identify problematic operators and suggest optimization strategies.

**Usage Example**:

```bash
# Install PTA (if not already installed)
pip install ascend-pta

# Run precision comparison between NPU and CPU results
pta compare --npu-result ./npu_output.npy --cpu-result ./cpu_output.npy --threshold 1e-5
```

For detailed usage, refer to the [PTA User Guide](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/ptatool/ptatool_000001.html).

#### 2. Ascend Debug Toolkit (ADK)

ADK provides low-level debugging capabilities for analyzing precision issues at the operator level.

**Key Features**:
- Operator-level precision comparison
- Intermediate tensor dumping and analysis
- Performance counter collection for precision-critical operations

Refer to the [ADK User Guide](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/debugtool/debugtool_000001.html) for detailed instructions.

### Common Precision Issue Patterns

| Issue Pattern | Possible Cause | Solution |
|---------------|----------------|----------|
| Large difference in loss values | Mixed precision configuration mismatch | Check `model.mixed_precision` settings in your YAML config |
| Gradient explosion/vanishing | Numerical instability in specific operators | Use operator-specific precision overrides in the configuration |
| Inconsistent inference results | Different random seeds or initialization | Ensure consistent random seeds across platforms |
| Slow convergence | Precision loss in accumulation operations | Enable high-precision accumulation for critical layers |

## Ascend Profiling Collection and Analysis

Profiling is essential for identifying performance bottlenecks in your training pipeline. This section provides guidance on how to collect and analyze profiling data on Ascend NPUs.

### Profiling Configuration

Before starting training, configure the following environment variables to enable profiling:

```bash
# Enable Ascend NPU profiling
export ASCEND_PROF_MODE=enable

# Configure profiling options
export ASCEND_PROF_OPTIONS=training_trace:task_trace:op_trace:aicpu_trace:mem_trace

# Set profiling output path
export PROFILING_DIR=./profiling_results
```

### Key Profiling Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| `training_trace` | Collect training-related events | `on`/`off` |
| `task_trace` | Collect task scheduling information | `on`/`off` |
| `op_trace` | Collect operator execution details | `on`/`off` |
| `aicpu_trace` | Collect AICPU operation information | `on`/`off` |
| `mem_trace` | Collect memory allocation information | `on`/`off` |

### Starting Profiling

Once the environment variables are set, start your training job as usual. The profiling data will be automatically collected during training.

### Analyzing Profiling Results

Ascend provides the **Ascend AI Profiler** tool for analyzing profiling data:

1. **Install Ascend AI Profiler**:

   ```bash
   pip install ascend-ai-profiler
   ```

2. **Generate Profile Report**:

   ```bash
   # Generate HTML report from profiling data
   profiler --input ./profiling_results --output ./profiling_report
   ```

3. **View Profile Report**:

   ```bash
   # Open the generated HTML report in a browser
   open ./profiling_report/index.html
   ```

### Key Metrics to Analyze

When analyzing the profiling report, focus on these key metrics:

1. **Computation Utilization**: Percentage of time the NPU computing units are active
2. **Memory Bandwidth Utilization**: Efficiency of memory access operations
3. **Kernel Execution Time**: Breakdown of time spent on each operator
4. **Data Transfer Overhead**: Time spent moving data between host and device
5. **Task Scheduling Overhead**: Time spent on task management and synchronization

### Performance Optimization Strategies

Based on profiling analysis, you can optimize your training pipeline using these strategies:

- **Operator Fusion**: Enable operator fusion to reduce kernel launch overhead
- **Memory Layout Optimization**: Adjust memory layout for better cache utilization
- **Parallelism Configuration**: Optimize FSDP, TP, and PP configurations
- **Data Pipeline Optimization**: Improve data loading and preprocessing efficiency

For more detailed information on profiling and optimization, refer to the [Ascend Profiling Guide](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/perftools/perftools_000001.html).

## FAQ

### Q: What should I do if I encounter "out of memory" errors?
A: Try reducing the `train.micro_batch_size` in the training command or enabling gradient checkpointing in the configuration file.

### Q: How do I speed up training on Ascend NPUs?
A: Ensure you're using the latest CANN version, enable mixed precision training, and adjust the CPU affinity configuration as described in the [Environment Variables](#environment-variables) section.

### Q: Can I train other models using a similar process?
A: Yes, most models follow a similar workflow. Refer to the specific model examples in the [examples](https://github.com/ByteDance-Seed/VeOmni/tree/main/docs/examples) directory for model-specific details.

### Q: Where can I find more information about VeOmni's features for Ascend NPUs?
A: Check the [VeOmni documentation](https://veomni.readthedocs.io/) and the [Ascend NPU specific documentation](https://github.com/ByteDance-Seed/VeOmni/tree/main/docs/hardware_support) for more details.

## Declarations

The Ascend support code, Dockerfile and image provided in the documentation are for reference only. If you intend to use them in a production environment, please contact the official channels. Thank you.
