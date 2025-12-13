# Ascend Quickstart

Last updated: 2025-11-28

We have added support for Huawei Ascend devices in VeOmni.

### Environment Requirements

| software  | version        |
| --------- | -------------- |
| Python    | >= 3.10, <3.12 |
| CANN      | == 8.3.RC1     |
| torch     | == 2.7.1       |
| torch_npu | == 2.7.1       |

Please refer to this [document](https://gitcode.com/Ascend/pytorch) for basic environment setup.

### Installing Dependencies with uv

#### 1. Enter the VeOmni root directory

    git clone https://github.com/ByteDance-Seed/VeOmni.git
    cd VeOmni

#### 2. Pin the Python version

    uv python pin 3.11

#### 3. (Optional) Set timeout

If the network is unstable, you can increase the timeout to avoid download failures by setting the UV_HTTP_TIMEOUT environment variable:

    export UV_HTTP_TIMEOUT=60

#### 4. Install the environment using uv

    uv sync --extra npu --allow-insecure-host github.com --allow-insecure-host pythonhosted.org

#### 5. Using the environment

After installation, a .venv folder will appear in the VeOmni project root. This is the environment created by uv.
Activate it with:

    source .venv/bin/activate

Check installed dependencies:

    uv pip list

### Quick Start

1.	Prepare the model and dataset.

2.	Set the NPROC_PER_NODE parameter in train.sh according to the number of available NPUs.

3.	Run the training script:

```bash
# Set environment variables
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2

bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml
```

Parallelism Support

| Feature          | Supported   |
| ---------------- | ----------- |
| fsdp             | ✅           |
| fsdp2            | ✅           |
| ulysses parallel | ✅           |
| expert_parallel  | In progress |