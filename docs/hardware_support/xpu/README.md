# VeOmni on Intel GPU (XPU)

This document describes how to run VeOmni on Intel GPUs, such as Arc Pro B-Series.

Intel GPU support ships in stock PyTorch: `torch.xpu` exposes the device and `xccl` provides the distributed communication backend. VeOmni selects both automatically when an Intel GPU is present and no CUDA, NPU or MLU device is, so the device-agnostic training path (FSDP2 sharding, Ulysses sequence parallel, checkpointing and profiling) applies as-is. Expert parallelism is not supported on XPU.

## Install

Install the XPU build of PyTorch, then VeOmni without the `gpu` extra (that extra pins the CUDA build of PyTorch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
pip install -e . transformers==5.16.1
```

Container images that already ship the XPU PyTorch stack, such as `vllm/vllm-openai-xpu`, work too; install VeOmni into them the same way.

## Configuration

Intel GPU runs every op on its `eager` backend: the per-op fields (`rms_norm_implementation`, `moe_implementation`, and so on) resolve to `eager` when left unset. Set the attention backend explicitly, since the default `flash_attention_2` has no Intel GPU build.:

```bash
--model.ops_implementation.attn_implementation sdpa
```

## Launch Training

`train.sh` auto-detects the accelerator in order: `nvidia-smi` → `rocm-smi` → `cnmon` → Intel GPU → NPU. On Intel GPU:

- device count comes from `torch.xpu.device_count()`;
- device visibility is controlled by `ZE_AFFINITY_MASK`, which plays the same role as `CUDA_VISIBLE_DEVICES` on GPU.

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path ${model_path} \
  --data.train_path  ${dataset_path} \
  --model.ops_implementation.attn_implementation sdpa
```

## Known Limitations

- **No fused kernels yet.** FlashAttention, Liger, fused MoE and the other CUDA kernels are not available; ops run on `eager` and attention on `sdpa`.
- **No expert parallelism.** Set `model.accelerator.ep_size=1` until an XPU-compatible implementation is available.
- **Low-precision ReduceScatter transport is CUDA-only.**
