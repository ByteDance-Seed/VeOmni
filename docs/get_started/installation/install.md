# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

VeOmni also supports other hardware platform, please refer to [Ascend](install_ascend.md).

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
wheels) / FA3 (sm90 abi3 prebuilt wheel) / FA4 / FlashQLA, diffusion / audio /
video / LoRA deps, and `megatron-energon` for the
optional energon dataset format. See
[pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml)
for the full list.

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

pip3 install -e .[gpu]
```
