# Ascend 950PR&950DT products Docker Image Build and Usage Guide

## Overview
This guide provides step-by-step instructions for building and using the Ascend 950PR&950DT products Docker image for VeOmni framework. The image is based on Huawei's Ascend CANN platform and includes all necessary dependencies for running multi-modal models on Ascend 950PR&950DT products accelerators.

## Prerequisites
- Docker installed on your system
- Access to Ascend 950PR&950DT products hardware accelerators
- Network access to pull the base image and install dependencies
- Proxy configuration (if required in your environment)

## Step 1: Pull the Base Image
First, pull the Huawei Ascend CANN base image. **Note: This image is for ARM64 architecture machines only.**

You can find the latest official Ascend CANN images at: [Ascend Hub](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

```bash
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.1-950-ubuntu22.04-py3.12
```

## Step 2: Build the Custom Image
Build the GDN-ready VeOmni Ascend 950PR&950DT products image using the provided Dockerfile. It includes `torch-npu==2.10.0.post7`, `triton-ascend`, and `fla_npu`. This ARM64 image uses pip and system Python; no virtual environment activation is required.

**Note:** Proxy settings are optional and only needed if your server requires proxy access to the internet. Remove the proxy arguments if not needed.

The GDN-ready Dockerfile also accepts `PIP_INDEX` and `APT_SOURCE` build arguments. `APT_SOURCE` must point to an Ubuntu Ports mirror.

```bash
# Optional proxy settings (remove if not needed)
docker build \
  --build-arg http_proxy=http://<user>:<pass>@<host>:<port> \
  --build-arg https_proxy=http://<user>:<pass>@<host>:<port> \
  --build-arg no_proxy=localhost,127.0.0.1 \
  -t ascend-a5-env:v1 \
  -f docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5 \
  .
```

Without proxy (simplified):
```bash
docker build \
  -t ascend-a5-env:v1 \
  -f docker/ascend/Dockerfile.ascend_9.2.1_torch_npu2.10.0.post7_a5 \
  .
```

### Image Components
The built image includes:
- Ubuntu 22.04 with Python 3.12
- Ascend CANN 9.2.1 runtime
- VeOmni framework with NPU support
- PyTorch 2.10.0 and torch-npu 2.10.0.post7
- Triton 3.6.0 and Triton Ascend 3.6.0 (base installed before the Ascend overlay)
- Ascend 950PR&950DT products FLA wheel built with `FLA_NPU_SOC=ascend950` from revision `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`
- Transformers 5.16.1; PyArrow 21.0.0; pandas 2.3.2
- TorchCodec built from v0.10.0; uv 0.11.16 available for separate CI environments
- TorchCodec for efficient video processing
- All necessary development tools and dependencies

## Step 3: Run the Container

### Basic Container Start

Use the image name from Step 2, `ascend-a5-env:v1`. The container instance is
named `ascend-a5-container`, matching the cleanup command in Step 5.
The following configuration uses paths present on the Ascend 950PR&950DT products host.
The runtime options follow the working Ascend 950PR&950DT products experiment command; private data
and cache mounts are omitted.

```bash
docker run --runtime=runc -it --net=host --ipc=host \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/ummu \
  --device=/dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/lib64/urma:/usr/lib64/urma:ro \
  -v /usr/lib64/liburma.so.0:/usr/lib64/liburma.so.0:ro \
  -v /usr/lib64/liburma_common.so.0:/usr/lib64/liburma_common.so.0:ro \
  -v /usr/lib64/libummu.so.1:/usr/lib64/libummu.so.1:ro \
  -v /usr/lib64/libnl-3.so.200:/usr/lib64/libnl-3.so.200:ro \
  -v /usr/lib64/libnl-genl-3.so.200:/usr/lib64/libnl-genl-3.so.200:ro \
  --name ascend-a5-container \
  ascend-a5-env:v1 /bin/bash
```

Ascend 950PR&950DT products uses `/dev/ummu` and `/dev/uburma` on this host; `/dev/devmm_svm` is absent.
The URMA provider directory and libraries accompany these devices. Mount stable
library names rather than copying versioned filenames from another machine:
`libummu.so.1` currently resolves to `libummu.so.1.0.4` on this host.
Verify every source path exists and library sources resolve to regular files.
Source the image's CANN environment before running Python to initialize
the toolkit and driver library search paths.

### Advanced Configuration Options

You can customize the container with the following optional configurations:

1. **Add more Ascend devices (example):** list each device explicitly. For
   eight-card training, expose devices 0 through 7.

   ```bash
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   ```

2. **Configure shared memory (example):** the basic command uses host IPC.
   To use private IPC instead, remove `--ipc=host` and specify a shared-memory
   size appropriate for the workload:

   ```bash
   --shm-size=64G \
   ```

3. **Add proxy environment variables (example):** replace the placeholders
   with your proxy settings if proxy access is required.

   ```bash
   -e http_proxy="http://<user>:<pass>@<host>:<port>" \
   -e https_proxy="http://<user>:<pass>@<host>:<port>" \
   -e no_proxy="localhost,127.0.0.1,.huawei.com" \
   ```

4. **Mount checkpoints (example):**

   ```bash
   -v /path/to/your/checkpoints:/app/ckpt:ro \
   ```

5. **Mount datasets (example):**

   ```bash
   -v /path/to/your/dataset:/app/dataset:ro \
   ```

6. **Mount HCCN configuration (example):** host networking is enabled in the
   basic command. Add this mount when communication uses the host configuration:

   ```bash
   -v /etc/hccn.conf:/etc/hccn.conf:ro \
   ```

7. **Run in the background (example):** replace `-it` with `-it -d`.
   Optionally add an automatic restart policy:

   ```bash
   --restart unless-stopped \
   ```

   Enter the running container with:

   ```bash
   docker exec -it ascend-a5-container /bin/bash
   ```

The basic command enables host networking and host IPC.
Model, dataset and cache mounts can be added according to the workload.

### Example: Complete Advanced Command

This eight-device example adds HCCN configuration and model/data mounts. Replace the model and dataset paths first.

```bash
docker run --runtime=runc -it --net=host --ipc=host \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/ummu \
  --device=/dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/lib64/urma:/usr/lib64/urma:ro \
  -v /usr/lib64/liburma.so.0:/usr/lib64/liburma.so.0:ro \
  -v /usr/lib64/liburma_common.so.0:/usr/lib64/liburma_common.so.0:ro \
  -v /usr/lib64/libummu.so.1:/usr/lib64/libummu.so.1:ro \
  -v /usr/lib64/libnl-3.so.200:/usr/lib64/libnl-3.so.200:ro \
  -v /usr/lib64/libnl-genl-3.so.200:/usr/lib64/libnl-genl-3.so.200:ro \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /path/to/your/checkpoints:/app/ckpt:ro \
  -v /path/to/your/dataset:/app/dataset:ro \
  --name ascend-a5-container \
  ascend-a5-env:v1 /bin/bash
```

The basic and advanced commands are alternatives; do not create both using the
same container name. The device and library paths follow the Ascend 950PR&950DT products host setup used by the experiment.

## Step 4: Run Training Inside the Container

Initialize CANN before running Python or training:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

The commands below are usage examples, not evidence of a full Ascend 950PR&950DT products UT pass.

### General Qwen3-VL Smoke Test

After starting the container with appropriate mounts, you can run training commands. Here's an example for Qwen3-VL training using generic paths:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path /app/ckpt/your-model-checkpoint \
    --data.train_path /app/dataset/your-dataset.json \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.max_seq_len 1024 \
    --train.global_batch_size 8
```

**Note:** Replace `/app/ckpt/your-model-checkpoint` and `/app/dataset/your-dataset.json` with the actual paths you used in your mount configuration.

### Qwen3.5 GDN Training Example

The GDN-ready image includes `triton-ascend` and `fla_npu`. Select the NPU
implementations explicitly when running Qwen3.5:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path /app/ckpt/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --model.ops_implementation.rms_norm_gated_implementation torch_npu \
    --model.ops_implementation.causal_conv1d_implementation triton \
    --model.ops_implementation.chunk_gated_delta_rule_implementation fla_npu \
    --train.max_steps 20
```

See the [Qwen3.5 training guide](../../examples/qwen3_5.md#start-training-on-npu)
for backend details and the Qwen3.5 MoE example.

## Step 5: Stop and Remove the Container
When you're done, stop and remove the container:

```bash
docker stop ascend-a5-container && docker rm ascend-a5-container
```

## Important Notes

### Device Access
The container requires access to all Ascend devices for proper functionality. The `--device` flags in the run command grant access to these devices.

### Mounts
- **Driver directories**: Required for Ascend runtime functionality
- **Checkpoints**: Mount pre-trained models to `/app/ckpt/`
- **Datasets**: Mount training data to appropriate locations
- **Shared memory**: Increase `--shm-size` for larger models or datasets

### Proxy Settings
Update the proxy settings in both the build and run commands to match your environment. Remove the proxy arguments if not needed.

### Dockerfile Details
The Dockerfile performs the following operations:
1. Sets up the Ubuntu 22.04 base with Ascend CANN
2. Configures system dependencies and development tools
3. Installs VeOmni framework with NPU support
4. Clones and builds TorchCodec for video processing
5. Sets up the working environment

## Troubleshooting

### Device Access Issues
- Ensure you have the correct permissions to access Ascend devices
- Verify the device paths exist on your host system
- Check that the Ascend driver is properly installed on the host

### Proxy Problems
- Verify proxy credentials and addresses are correct
- Ensure the proxy allows access to required domains
- Try removing proxy settings if running in an internal network

### Build Failures
- Check network connectivity for pulling dependencies
- Ensure sufficient disk space is available
- Review the full build log for specific error messages

## Support
For additional help, please refer to:
- VeOmni documentation
- Ascend CANN documentation
- Docker documentation for container management
