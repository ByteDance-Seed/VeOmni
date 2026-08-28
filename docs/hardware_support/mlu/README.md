# VeOmni on Cambricon MLU

This document describes how to run VeOmni on Cambricon MLU.

VeOmni now supports MLU-specific kernel code in its OSS tree. On MLU it runs through PyTorch's `torch_mlu` backend: `torch.mlu` exposes the device and `cncl` provides the distributed communication backend. Most of the device-agnostic training path (FSDP2 sharding, Ulysses sequence parallel, expert parallel, checkpointing, profiling) applies as-is.

## Supported Models

VeOmni supports a wide range of models on Cambricon MLU, across modalities (text / VLM / Omni / DiT) and architectures (dense / MoE+EP), including llama / qwen3 / qwen3_vl / qwen3.5 / qwen-image / wan2.1.

The MoE expert layer is accelerated by the `fused_moe_kernel`. On MLU there are two interchangeable `moe_implementation` backends. The faster one is **`fused_mlu`**, which requires `torch_mlu` plus `apex` (specifically `apex.contrib.grouped_gemm`) and runs the Apex grouped-GEMM kernel for better performance. When `apex` is not available, you can use **`fused_mlu_triton`** instead: it requires only `torch_mlu` plus `triton` and runs the Triton grouped-GEMM kernel with no apex dependency. Both backends cover the same MoE path; pick `fused_mlu` for speed when the apex stack is present, otherwise `fused_mlu_triton` runs everywhere `triton` does.

Other ops on MLU (RMSNorm, RoPE, SwiGLU, load balancing loss, cross entropy loss) currently fall back to the `eager` (huggingface reference) implementation; more fused kernels will be supported soon for better performance.

## Get Started

### 1. Pull the Base Image

Please contact Cambricon engineer to obtain the cambricon_release docker image, which ships the validated MLU stack (PyTorch + `torch_mlu`, `apex` with `apex.contrib.grouped_gemm`, `triton`, `flash_attn`, `transformers`, etc.).

Example (start a container):

The image does **not** contain VeOmni itself — clone it yourself and mount it into the container:

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
```

```bash
docker_image=cambricon_release_image
docker_name=veomni_test
docker run -itd \
    --name ${docker_name} \
    --network=host \
    --ipc=host \
    --pid=host \
    --shm-size 512G \
    --device /dev/cambricon_ctl \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    ${docker_image} \
    /bin/bash

docker exec -it veomni_test /bin/bash
```

Once inside, register VeOmni as an editable package, and VeOmni supports two installation methods for mlu: `uv` (recommended for faster installation) and `pip`:

```bash
uv pip install -e .
pip install -e .
```


## Launch Training

`train.sh` auto-detects the accelerator in order: `nvidia-smi` → `rocm-smi` → `cnmon` → NPU. On MLU it takes the MLU branch:

- device count is detected via `cnmon`;
- device visibility is controlled by `MLU_VISIBLE_DEVICES`, which plays the same role as `CUDA_VISIBLE_DEVICES` on GPU.

Example (real-model SFT, from `docs/examples/qwen3.md`):

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path ${model_path} \
  --data.train_path  ${dataset_path}
```

`train.sh` ultimately launches training through `torchrun`; all FSDP2 / EP / SP / checkpoint settings come from the config file, exactly as on GPU.


## Known Limitations

- **CUDA-only kernels fall back.** (FA3/FA4/Quack/FlashMLA/DSA, e.g. `gpt_oss`) fall back to triton/eager on MLU, or are skipped.
- **MoE speed depends on apex.** `fused_mlu` (apex grouped-GEMM) is the faster path; without `apex` the `fused_mlu_triton` Triton path is used instead.
