(installation-install_ascend)=

# Installation with Ascend NPU

## Required Environment

CANN == 8.3.RC1

## Prepare CANN

Choose one of the following methods to use CANN:

1. Install CANN according to the [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)

2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Install with uv or pip

**UV**

> Recommend to use [uv](https://docs.astral.sh/uv/) for faster and easier installation.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

# use the locked uv env
uv sync --locked  --extra npu
source .venv/bin/activate
```

You can use `--extra` to install other optional dependencies. Refer to [pyproject.toml](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml) for more details.

```bash
# eg. install with video processing dependencies (torchcodec, PyAV, librosa)
uv sync --locked  --extra npu --extra video
```

> **Note**: For video processing with the `video` extra, you also need to install ffmpeg separately:
> ```bash
> # Ubuntu/Debian/openEuler
> sudo apt-get install ffmpeg
> # or
> sudo yum install ffmpeg
> ```

**Pip**

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

pip3 install -e .[npu]
```

## Ascend relevant Environment variables

```bash
# Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
