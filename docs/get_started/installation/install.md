# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

VeOmni also supports other hardware platforms. See the installation guides for
[Ascend x86](install_ascend_x86.md), [Ascend ARM](install_ascend_arm.md), and
[AMD ROCm](../../hardware_support/rocm/README.md).

## Required Environment

CUDA 13.0 (the `gpu` extra targets `+cu130` torch wheels and the `nvcr.io/nvidia/pytorch:25.11-py3` base image).

## Install with uv or pip

**UV**

> Recommend to use [uv](https://docs.astral.sh/uv/) for faster and easier installation.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

uv sync --locked --extra gpu
source .venv/bin/activate
```

`gpu` is a single full superset: cu130 torch, FA2 (cp311/cp312 prebuilt
wheels) / FA3 (sm90 abi3 prebuilt wheel) / FA4 / FlashQLA, DeepGEMM, diffusion / audio /
video / LoRA deps, and `megatron-energon` for the
optional energon dataset format. See
[pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml)
for the full list.

DeepGEMM wheels cover Python 3.11/3.12 on Linux x86_64/aarch64 and require
glibc >=2.38 plus a compatible libstdc++ (Ubuntu 24.04 / NGC 25.11).
FP8 MoE remains opt-in with `moe_implementation: deepgemm_fp8` and requires
Hopper SM90 or Blackwell SM100. Runtime GEMM JIT still needs NVCC and a C++20
compiler; see [FP8 MoE training](../../design/deepgemm_fp8_moe.md).

### Optional MagiAttention extra

MagiAttention is an optional NVIDIA SM100+ extra. It source-builds CUDA
extensions, so omit it on older GPUs and CPU environments.
Install it together with the `gpu` extra:

```bash
uv sync --locked --extra gpu --extra magi
```

MagiAttention uses CUTE DSL/JIT on SM100 and newer GPUs. On SM90, select
FlexAttention instead; its FA4 backend is included in the `gpu` extra.

> **Note**: video/audio processing also needs ffmpeg installed at the OS level:
> ```bash
> # Ubuntu/Debian
> sudo apt-get install ffmpeg
>
> # macOS
> brew install ffmpeg
> ```

**Pip**

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

pip3 install -e .[gpu] \
  --find-links https://verl-project.github.io/verl-wheelhouse/cu130/torch2.11/simple/deep-gemm/
```

Pip does not read `[tool.uv.sources]`; the wheelhouse link supplies the pinned
DeepGEMM wheel for the matching torch 2.11 / CUDA 13 environment.
To include MagiAttention, use `.[gpu,magi]` in the same command.
