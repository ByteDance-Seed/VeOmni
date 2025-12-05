# Ascend Quickstart

## Installing CANN

### Choose one of the following methods to install CANN:

1. Install CANN according to the  [official documentation](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)
2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Installing VeOmni Dependencies with uv

### 1. Enter the VeOmni root directory

    cd VeOmni

### 2. Install the environment using uv

    uv sync --extra npu

### 3. Using the environment

    source .venv/bin/activate

## Ascend relevant Environment variables

```shell
    source $CANN_path/ascend-toolkit/set_env.sh
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
## Quick Start Training

You're successfully completed the environment setup. Now, you're ready to start training your model.

> **Please refer to [Qwen3 VL Quickstart Guide](../examples/qwen3_vl.md) for detailed instructions**.
