# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

VeOmni also supports other hardware platform, please refer to [Ascend NPU Installation](../installation/install_ascend_x86.md).

## Required Environment

CUDA == 12.8

## Supported Hardware and Operating Systems

| Hardware | Operating System | Architecture |
|----------|-----------------|-------------|
| Nvidia GPU (CUDA 12.8) | Ubuntu 22.04 / 24.04 | x86_64 |
| Ascend A2 (910B) | Ubuntu 22.04 | x86_64 / ARM64 |
| Ascend A3 (910B) | Ubuntu 22.04 | x86_64 / ARM64 |
| Ascend 950 (A5) | Ubuntu 22.04 | x86_64 / ARM64 |

> **Note**: For detailed information about Ascend NPU hardware support, please refer to [Get Started with Ascend NPU](../hardware_support/get_started_npu.md).
> For scenarios pending further validation on A5, please refer to [A5 Features Pending Validation](../hardware_support/a5_unsupported_features.md).

## Install with uv or pip

**UV**

> Recommend to use [uv](https://docs.astral.sh/uv/) for faster and easier installation.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

# use the locked uv env
uv sync --locked  --extra gpu
source .venv/bin/activate
```

You can use `--extra` to install other optional dependencies. Refer to [pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml) for more details.

```bash
# eg. install with dit dependencies in GPU
uv sync --locked  --extra gpu --extra dit

# eg. install with video/audio processing dependencies (torchcodec, PyAV, librosa, soundfile)
# Note: `video` and `audio` extras are equivalent - both include video and audio processing
uv sync --locked  --extra gpu --extra video
# or equivalently:
uv sync --locked  --extra gpu --extra audio
```

> **Note**: For video/audio processing with the `video` or `audio` extra, you also need to install ffmpeg separately:
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
