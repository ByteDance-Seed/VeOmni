# Ascend Quickstart

## Installing CANN

### Choose one of the following methods to install CANN:

1. Install CANN according to the  [official documentation](https://www.hiascend.com/document/detail/en/canncommercial/800/softwareinst/instg/instg_0008.html)
2. Download and use [the CANN image](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)

## Installing VeOmni Dependencies with uv

### 1. Enter the VeOmni root directory

```shell
cd VeOmni
```

### 2. Install the environment using uv

```shell
uv sync --extra npu
# If you encounter errors, try running the following command:
# uv sync --extra npu --allow-insecure-host github.com --allow-insecure-host pythonhosted.org
```

### 3. Using the environment
```shell
source .venv/bin/activate
```
## Ascend relevant Environment variables

```shell
# Make sure CANN_path is set to your CANN installation directory, e.g., export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## All-in-one

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
