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

2026/5/11: VeOmni provides images of the version of Ascend Cann9.0.0.

2025/12/23: VeOmni supports training on Ascend NPU.

## Installation

VeOmni supports two installation methods for Ascend NPUs: `uv` (recommended for faster installation) and `pip`. Note that ARM architecture machines only support `pip` installation.

### Installation Options

- **x86 Architecture**: Supports both `uv` and `pip` installation methods
- **ARM Architecture**: Only supports `pip` installation method

### Detailed Installation Guide

Please refer to the specific installation guides based on your architecture:

- [Installation with Ascend NPU (x86)](../get_started/installation/install_ascend_x86.md)
- [Installation with Ascend NPU (ARM)](../get_started/installation/install_ascend_arm.md)

### Docker Support

VeOmni also provides Docker support for Ascend NPUs. For detailed instructions on building and using Ascend Docker images, please refer to:

- [Ascend A3 Docker Image Build and Usage Guide](./AscendDockerUsage/build_a3_docker.md)
- [Ascend A2 Docker Image Build and Usage Guide](./AscendDockerUsage/build_a2_docker.md)

## Version Compatibility

The following table shows the supported software versions for VeOmni when running on Ascend NPUs:

| VeOmni Version | PyTorch | torch_npu | CANN Version | Python Version |
|----------------|-------- | -----------|--------------|----------------|
| 0.1.0 | 2.7.1                | 2.7.1             | 8.3rc2/9.0.0      | 3.11           |
| main  | In-development    | In-development | In-development | In-development |

## Supported Models

VeOmni supports a wide range of models on Ascend NPUs, including large language models, multimodal models, and diffusion models. Below is a comprehensive list of supported models with their features:

| Model                | Model Size       | Support | FSDP1 | FSDP2 | EP | SP | Note                                           |
|----------------------|------------------|---------|-------|-------|----|----|------------------------------------------------|
| [Qwen3](../examples/qwen3.md) | 8B              | ✅       |       | ✅     |    | ✅   |
|                      | 30B               | ✅       |       | ✅     | ✅    | ✅   |
| [Qwen3.5](../examples/qwen3.md) | 9B    |          |         | ✅    |      |✅    | supporting   |
|                      | 35B-A3B              |         |       | ✅     |✅    |✅    |  supporting                                   |
| [Qwen3-VL](../examples/qwen3_vl.md) | 8B               | ✅       |       | ✅     |    | ✅  |                               |
|                      | 30B              | ✅       |       | ✅     | ✅  | ✅  |                                                |
| [Wan2.1](../examples/wan2.1.md)    | 1.3B              | ✅       | ✅     |       |    | ✅  | prototype                              |
| [Qwen3Omni](../examples/qwen3_omni_moe.md)    | 30B              | ✅       |   | ✅        |    | ✅  | prototype                              |

**Legend:**
- **FSDP1**: Fully Sharded Data Parallel version 1
- **FSDP2**: Fully Sharded Data Parallel version 2 (recommended)
- **EP**: Expert Parallel - for MoE models
- **SP**: Sequence Parallel - enables longer sequence training

For detailed configuration files and training examples, please refer to the [configs](https://github.com/ByteDance-Seed/VeOmni/tree/main/configs) directory in the repository.

For information about optimizing environment variables for Ascend NPUs, please refer to our dedicated documentation:

[Ascend Environment Variables Configuration Guide](npu_variables.md)

## Typical Usage

For a complete step-by-step guide on training the Qwen3-VL 8B model on Ascend NPUs, including dataset preparation, model configuration, training, and checkpoint management, please refer to our dedicated documentation:

[Typical Usage: Qwen3-VL 8B Training on Ascend NPU](typical_usage.md)

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
