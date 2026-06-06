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

# use the locked uv env
uv sync --locked --extra gpu
source .venv/bin/activate
```

The `gpu` extra is a single, full superset: it pulls in the cu130 torch
stack, every attention kernel (FA2 / FA3 / FA4 / FlashQLA — all source-built),
plus the diffusion / audio / video / RLHF / LoRA Python deps. There is no
need to chain multiple `--extra` flags any more — the older `audio`, `video`,
`dit`, `trl`, `lora`, `fa3`, `fa4`, `flash-qla` extras have been folded into
`gpu`. A first sync is slow (the source builds for FA2/FA3/FA4/FlashQLA take
~60–90 min combined); uv caches the built wheels under `~/.cache/uv` so
subsequent syncs are fast. See [pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml) for the exact dependency list.

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
