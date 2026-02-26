# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

VeOmni also supports other hardware platform, please refer to [Ascend](install_ascend.md).

## Required Environment

CUDA == 12.8

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
