# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

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
```

**Pip**

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

pip3 install -e .[gpu]
```
