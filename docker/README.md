# A100 training container (CUDA 12.6)

This container is the compatibility-focused stack for the current 8 x
A100-SXM4 host:

- VeOmni commit `10adfab65975a4d6d449f860996e206499cf3132`
- NVIDIA NGC PyTorch `24.08-py3`
- Ubuntu 22.04, Python 3.10, and the CUDA 12.6 compiler/toolkit
- PyTorch 2.7.1 + CUDA 12.6 runtime, torchvision 0.22.1, torchaudio 2.7.1,
  and Triton 3.3.1
- Transformers 4.51.3
- FlashAttention 2.7.4.post1, using the upstream CPython 3.10 / PyTorch 2.7 /
  CUDA 12 / cxx11abiTRUE wheel
- FlexAttention from PyTorch 2.7.1 through the Transformers 4.51.3 integration

The local Qwen3-0.6B checkpoint itself records Transformers 4.51.0, and the
selected VeOmni commit is the upstream fix for Qwen3 gradient checkpointing on
that 4.51 generation.

The Dockerfile also pins the NGC base manifest digest, so a future mutable tag
cannot silently change the CUDA toolchain. It disables NGC's inherited
`pypi.ngc.nvidia.com` extra index and selects each package index explicitly,
so dependency sources are not silently mixed.

The base image's DALI/ModelOpt/Torch-TensorRT extras are removed because they
are outside this text-training scope and retain pins or malformed metadata from
the original nightly Torch stack. Its overlapping `pytorch-triton` package is
also removed before Triton 3.3.1 is reinstalled. Ninja 1.11.1.1 is only a
bootstrap requirement and is replaced in the Dockerfile by 1.11.1.4, whose
wheel metadata passes `pip check`.

FlashAttention 2.6.3 predates stable PyTorch 2.5 wheels and falls back to a
local source build that unconditionally emits both sm80 and sm90 code with a
CUDA 12.6 compiler. Version 2.7.4.post1 has an official wheel matching this
exact Python/PyTorch/ABI tuple. Its release-asset SHA-256 is pinned in the
Dockerfile, and the runtime smoke test executes BF16 FlashAttention on every
physical A100 rather than treating a successful import as sufficient.

FlexAttention is part of PyTorch, not a separate package. PyTorch 2.5 was its
first prototype release. Transformers 4.51.3 contains an explicit workaround
for a PyTorch 2.6 training compilation issue, while its normal PyTorch 2.7 path
uses the repaired compiler flow. The FlexAttention smoke test covers causal
BlockMask, score modification, GQA, BF16 forward/backward, a non-128-multiple
sequence length, and comparison against SDPA on every A100. The historical
VeOmni argument whitelist is extended to expose the FlexAttention path already
implemented by its Qwen3 model.

It is designed for the installed `560.35.03` driver and does not require any
host driver, Fabric Manager, kernel, or operating-system change. R560 supports
the CUDA 12.6 runtime embedded in the PyTorch wheels and the CUDA 12.6 compiler
toolchain in the image.

The image deliberately does not install Transformers 4.56 or vLLM. Training
configs may use either `attn_implementation: flash_attention_2` or
`attn_implementation: flex_attention`.

This image is validated for the local Qwen3 training target. The checkout's
optional Seed-OSS module requires the newer `transformers.masking_utils` and
`transformers.models.seed_oss` APIs, which do not exist in Transformers 4.51.3;
VeOmni therefore logs and skips that optional registry entry. Use a separate
newer-stack image if Seed-OSS becomes a target instead of silently upgrading
this Qwen3 environment.

NGC's startup banner is baked into the base image and still prints its original
PyTorch 2.5 nightly version. It is not the installed runtime version; the build
and smoke checks assert and print PyTorch `2.7.1+cu126` from Python.

## 1. Activate Docker group membership

Joining the group only affects new login sessions. Either log out and back in,
or run this once in the shell that will build the image:

```bash
newgrp docker
```

Confirm with:

```bash
docker version
```

## 2. Pull and build

The NGC image is public, but its first download is large. A failed pull can be
repeated; Docker resumes completed layers.

```bash
docker pull nvcr.io/nvidia/pytorch:24.08-py3
cd /mgData4/fhb/code/training/VeOmni
git switch compat/qwen3-a100-cu126
./docker/build.sh
```

An alternative package index can be selected at build time without editing the
Dockerfile:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple \
  -f docker/Dockerfile \
  -t veomni:a100-cu126-torch271-tf451 .
```

Do not put credentials in `PIP_INDEX_URL`; Docker build metadata may retain the
argument.

`build.sh` automatically verifies and serves the canonical cached wheel at
`/mgData4/datasets/cache/veomni-docker/wheels/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
over a temporary loopback HTTP server. The server is stopped after the build.
If that file is absent, the Dockerfile downloads the same release asset from
GitHub. `VEOMNI_FLASH_ATTN_WHEEL` can select another local copy, while
`VEOMNI_FLASH_ATTN_WHEEL_URL` can select a trusted remote mirror. The pinned
upstream SHA-256 is verified in every case, so a mismatched payload is rejected.
If loopback port 18765 is already occupied, set `VEOMNI_WHEEL_SERVER_PORT` to
another unprivileged port.

## 3. Verify the complete stack

Run CUDA and FlashAttention 2 on every A100:

```bash
./docker/run.sh python docker/smoke_test.py
```

Run compiled FlexAttention forward/backward on every A100:

```bash
./docker/run.sh python docker/flex_smoke.py
```

Then exercise an NCCL all-reduce across all eight GPUs:

```bash
./docker/run.sh \
  torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29617 docker/nccl_smoke.py
```

Finally, run one real Qwen3-0.6B optimizer step through each backend with the
small local fixture. Both configs enable gradient checkpointing so the selected
VeOmni compatibility fix is exercised:

```bash
./docker/run.sh env CUDA_VISIBLE_DEVICES=0 \
  torchrun --nnodes=1 --node-rank=0 --nproc-per-node=1 \
  --master-addr=127.0.0.1 --master-port=29618 \
  tasks/train_torch.py docker/qwen3_06b_smoke.yaml

./docker/run.sh env CUDA_VISIBLE_DEVICES=0 \
  torchrun --nnodes=1 --node-rank=0 --nproc-per-node=1 \
  --master-addr=127.0.0.1 --master-port=29619 \
  tasks/train_torch.py docker/qwen3_06b_flex_smoke.yaml
```

These commands write only test metadata below the canonical run directories
`/mgData4/ckpts/veomni_docker_qwen3_0_6b_smoke` and
`/mgData4/ckpts/veomni_docker_qwen3_0_6b_flex_smoke`; checkpoint saving and W&B
are disabled.

Run the 50,000-row canonical Dolci subset for a 10-step, eight-GPU Qwen3-0.6B
SFT validation through FlashAttention 2:

```bash
./docker/run.sh \
  torchrun --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29620 \
  tasks/train_torch.py docker/qwen3_06b_dolci_sft_10step.yaml
```

This writes run metadata and tokenizer assets to
`/mgData4/ckpts/veomni_docker_qwen3_0_6b_dolci_sft_10step_fa274`. It deliberately
does not save a model checkpoint or use W&B; the run validates data loading,
ChatML masking, distributed training, gradient checkpointing, FlashAttention 2,
backpropagation, and optimizer updates without retaining a throwaway model.

## 4. Enter the training environment

```bash
./docker/run.sh
```

The launcher:

- exposes all GPUs and the host network;
- uses host IPC and training-safe memlock/stack limits;
- mounts `/mgData4` at the identical path inside the container;
- bind-mounts this checkout at `/workspace/VeOmni`;
- runs as the calling host UID/GID so checkpoints are not owned by root;
- persists Hugging Face, extension, and user caches below the canonical
  `/mgData4/datasets/cache/veomni-docker` cache directory.

Any command can be passed directly, for example:

```bash
./docker/run.sh python -c \
  'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

Override the image name, data root, or canonical cache root with
`VEOMNI_IMAGE`, `VEOMNI_DATA_ROOT`, and `VEOMNI_CACHE_ROOT` if required.
