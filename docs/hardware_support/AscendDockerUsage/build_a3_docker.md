# Ascend A3 Docker Image Build and Usage Guide

## Overview
This guide provides step-by-step instructions for building and using the Ascend A3 Docker image for VeOmni framework. The image is based on Huawei's Ascend CANN platform and includes all necessary dependencies for running multi-modal models on Ascend A3 accelerators.

## Prerequisites
- Docker installed on your system
- Access to Ascend A3 hardware accelerators
- Network access to pull the base image and install dependencies
- Proxy configuration (if required in your environment)

## Step 1: Pull the Base Image
First, pull the Huawei Ascend CANN base image. **Note: This image is for ARM64 architecture machines only.**

You can find the latest official Ascend CANN images at: [Ascend Hub](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

```bash
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc2-a3-ubuntu22.04-py3.11
```

## Step 2: Build the Custom Image
Build the VeOmni Ascend A3 image using the provided Dockerfile.

**Note:** Proxy settings are optional and only needed if your server requires proxy access to the internet. Remove the proxy arguments if not needed.

```bash
docker build \
  # Optional proxy settings (remove if not needed)
  --build-arg http_proxy=http://<user>:<pass>@<host>:<port> \
  --build-arg https_proxy=http://<user>:<pass>@<host>:<port> \
  --build-arg no_proxy=localhost,127.0.0.1 \
  -t ascend-a3-env:v1 \
  -f docker/ascend/Dockerfile.ascend_8.3rc2_a3 \
  .
```

Without proxy (simplified):
```bash
docker build \
  -t ascend-a3-env:v1 \
  -f docker/ascend/Dockerfile.ascend_8.3rc2_a3 \
  .
```

### Image Components
The built image includes:
- Ubuntu 22.04 with Python 3.11
- Ascend CANN 8.3.rc2 runtime
- VeOmni framework with NPU support
- TorchCodec for efficient video processing
- All necessary development tools and dependencies

## Step 3: Run the Container

### Basic Container Start
Start the container with Ascend device access. The example below uses a wildcard to include all Ascend cards, but you can also specify individual devices if needed:

```bash
docker run --runtime=runc -it \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci* \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  --name ascend-a3-container \
  ascend-a3-env:v1 \
  /bin/bash
```

### Advanced Configuration Options
You can enhance the basic command with the following optional configurations:

1. **Add more Ascend devices** by either listing them individually or using a wildcard to include all cards matching the naming pattern:
   ```bash
   # Option 1: List individual devices
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \

   # Option 2: Use wildcard to include all davinci devices
   --device=/dev/davinci* \
   ```

2. **Increase shared memory** (recommended for larger models):
   ```bash
   --shm-size=64G \
   ```

3. **Add proxy environment variables** (if needed):
   ```bash
   -e http_proxy="http://<user>:<pass>@<host>:<port>" \
   -e https_proxy="http://<user>:<pass>@<host>:<port>" \
   -e no_proxy="localhost,127.0.0.1,.huawei.com" \
   ```

4. **Mount checkpoints** (example):
   ```bash
   -v /path/to/your/checkpoints:/app/ckpt/:ro \
   ```

5. **Mount datasets** (example):
   ```bash
   -v /path/to/your/dataset.json:/app/dataset/dataset.json:ro \
   -v /path/to/your/images:/app/dataset/images:ro \
   ```

### Example: Complete Advanced Command
Here's an example combining all these options:

```bash
docker run --runtime=runc -it \
  --ulimit nproc=65535 \
  --ulimit nofile=65535 \
  --device=/dev/davinci* \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --shm-size=64G \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/tools:/usr/local/Ascend/driver/tools:ro \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro \
  -v /path/to/your/checkpoints:/app/ckpt/:ro \
  -v /path/to/your/dataset:/app/dataset/:ro \
  --name ascend-a3-container \
  ascend-a3-env:v1 \
  /bin/bash
```

## Step 4: Run Training Inside the Container
After starting the container with appropriate mounts, you can run training commands. Here's an example for Qwen3-VL training using generic paths:

```bash
bash train.sh tasks/deprecated_task/train_qwen_vl.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path /app/ckpt/your-model-checkpoint \
    --data.train_path /app/dataset/your-dataset.json \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.max_seq_len 1024 \
    --train.global_batch_size 8
```

**Note:** Replace `/app/ckpt/your-model-checkpoint` and `/app/dataset/your-dataset.json` with the actual paths you used in your mount configuration.

## Step 5: Stop and Remove the Container
When you're done, stop and remove the container:

```bash
docker stop ascend-a3-container && docker rm ascend-a3-container
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
