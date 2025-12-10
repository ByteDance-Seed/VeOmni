# Quickstart

## Installing VeOmni Dependencies with uv

### Enter the VeOmni root directory

```shell
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
```

### Install the environment using uv

For GPU:
```shell
uv sync --extra gpu
```

For Ascend NPU
```shell
uv sync --extra npu
# If you encounter errors, try running the following command:
# uv sync --extra npu --allow-insecure-host github.com --allow-insecure-host pythonhosted.org
```

You can install other optional deps by adding --extra like --extra dit

### Using the environment

```shell
source .venv/bin/activate
```

## Installing CANN (Only Ascend NPU device need)

### Choose one of the following methods to install CANN:

1. Install CANN according to the [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Ascend relevant Environment variables (Only Ascend NPU device need)

```shell
# Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## All-in-one For GPU
```shell
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
uv sync --extra gpu
source .venv/bin/activate
```

## All-in-one For NPU

```shell
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
uv sync --extra npu
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## Quick Start Training

You're successfully completed the environment setup. Now, you're ready to start training your model.

> **Please refer to [Qwen3 VL Quickstart Guide](../examples/qwen3_vl.md) for detailed instructions**.
