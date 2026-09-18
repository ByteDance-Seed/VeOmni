# Installation with Nvidia GPU

In this section, we provide the installation guide for Nvidia GPU.

VeOmni also supports other hardware platforms. See the installation guides for
[Ascend x86](install_ascend_x86.md), [Ascend ARM](install_ascend_arm.md), and
[AMD ROCm](../../hardware_support/rocm/README.md).

## Required Environment

Use Linux and Python 3.11 or 3.12, as declared in `pyproject.toml`.
CUDA 13.0 (the `gpu` extra targets `+cu130` torch wheels and the `nvcr.io/nvidia/pytorch:25.11-py3` base image).

## Install with uv (recommended)

Use [uv](https://docs.astral.sh/uv/) in the version range declared in
`pyproject.toml`. The lockfile includes the accelerator packages and the
default dependency groups, including the supported Transformers pin.

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

### Optional MagiAttention extra

MagiAttention is an optional NVIDIA SM90+ extra. It source-builds CUDA
extensions, so omit it on Ampere/Ada (A100/L20) and CPU environments.
Install it together with the `gpu` extra:

```bash
uv sync --locked --extra gpu --extra magi
```

MagiAttention uses CUTE DSL/JIT on SM100 and newer GPUs. SM90 GPUs require an additional CUTLASS overlay after the Magi extra is synced:

```bash
bash scripts/kernel/install_magi_sm90.sh
```

The verified default enables BF16/FP16 inputs, the hdim128 bucket, and nfunc 1/3/5. Use `--help` to inspect optional build overrides. A later exact `uv sync` without `--extra magi` can remove the overlay, so rerun the installer before using MagiAttention on SM90.

> **Note**: video/audio processing also needs ffmpeg installed at the OS level:
> ```bash
> # Ubuntu/Debian
> sudo apt-get install ffmpeg
>
> # macOS
> brew install ffmpeg
> ```

## Using pip in an existing environment

The commands below assume the matching accelerator dependencies are already
installed, for example in a prepared container. Pip does not consume
`tool.uv.sources`, the uv dependency groups, or `uv.lock`; a bare
`pip install -e '.[gpu]'` does not reproduce the uv environment. Use the uv
workflow above when creating a new environment.

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni

python -m pip install -e . --no-deps
python -m pip install transformers==5.16.1
```

The supported Transformers version is also recorded in `pyproject.toml`.
For MagiAttention, use the optional uv extra described above.

## Next steps

- Complete the [Quick Start](../quick_start.md), then explore the
  [Qwen3 training guide](../../examples/qwen3.md).
- For code or documentation contributions, read the
  [contribution guide](https://github.com/ByteDance-Seed/VeOmni/blob/main/CONTRIBUTING.md).
