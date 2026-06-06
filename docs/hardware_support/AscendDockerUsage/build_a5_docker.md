# Ascend 950 (A5) Docker Image Build and Usage Guide

## Overview
This guide provides step-by-step instructions for building and using the Ascend 950 (A5) Docker image for VeOmni framework. The A5 product is part of the Ascend 950 series and supports both x86_64 and ARM64 architectures.

> **Note**: For Ascend A2 (910B) Docker images, please refer to [Ascend A2 Docker Guide](../build_a2_docker.md).
> **Note**: The A5 Docker image is currently under preparation and will be available soon.

## Prerequisites
- Docker installed on your system
- Access to Ascend 950 (A5) hardware accelerators
- Network access to pull the base image and install dependencies
- Proxy configuration (if required in your environment)

## Supported Base Image

The A5 Docker image is based on the following CANN base image (to be released):

```bash
# for x86
docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-950-ubuntu22.04-py3.11

# for arm64
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-950-ubuntu22.04-py3.11
```

> **Note**: The base image tag for A5 will be updated once published by the Ascend platform team. Please check back for the latest image.

## Build and Run Instructions

Build and run instructions for A5 will be added once the Docker image is ready. The A5 Docker image will follow the same structure as the A2/A3 images, with appropriate CANN 9.0.0 base and VeOmni dependencies pre-installed.

## Operator Configuration for A5

When training on A5, use the following operator configuration in your YAML:

```yaml
model:
  ops_implementation:
    attn_implementation: "sdpa"
    moe_implementation: "fused_npu"
    cross_entropy_loss_implementation: "npu"
    rms_norm_implementation: "npu"
    rotary_pos_emb_implementation: "npu"
    swiglu_mlp_implementation: "eager"
    load_balancing_loss_implementation: "eager"
    rms_norm_gated_implementation: "eager"
```

> **Note**: Do not use `flash_attention_2` or `liger_kernel` backends on A5. See [A5 Features Pending Validation](../a5_unsupported_features.md) for details.
