# VeOmni on Cambricon MLU

This document describes how to run VeOmni on MLU.

## Get Started

VeOmni supports a wide range of models on Cambricon MLU, across modalities (text / VLM / Omni / DiT) and architectures (dense / MoE+EP), including llama / qwen3 / qwen3_vl / qwen3.5 / qwen-image / wan2.1. We have supported `fused_mlu`  for `fused_moe_kernel`, and more fused kernel will be supported for better performance.

### 1. Pull the Base Image

Please contact Cambricon engineer to get the cambricon_release docker images.

Example (start a container):

```bash
docker_image=cambricon_release_image
docker_name=veomni_test
docker run -itd \
    --name ${docker_name} \
    --privileged=true \
    --network=host \
    --ipc=host \
    --pid=host \
    --shm-size 512G \
    --device /dev/cambricon_ctl \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    -w /workspace \
    ${docker_image} \
    /bin/bash

docker exec -it veomni_test /bin/bash
```

Install VeOmni in MLU: 

```bash
pip install -e .
```

### 2. Launch training

`train.sh` auto-detects the accelerator in order: `nvidia-smi` → `rocm-smi` → `cnmon` → NPU. On MLU it takes the MLU branch:

- device count is detected via `cnmon`;
- device visibility is controlled by `MLU_VISIBLE_DEVICES`.

Example (from `docs/examples/qwen3.md`):

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path ${model_path} \
  --data.train_path  ${dateset_path}
```
