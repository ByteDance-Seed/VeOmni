# Ascend Quickstart

Last updated: 11/27/2025

我们在VeOmni上增加对华为昇腾设备的支持。

### 基础环境准备

+-----------+--------------------------+
| software  | version                  |
+-----------+--------------------------+
| Python    | >= 3.10, <3.12           |
+-----------+--------------------------+
| CANN      | == 8.3.RC1               |
+-----------+--------------------------+
| torch     | == 2.7.1                 |
+-----------+--------------------------+
| torch_npu | == 2.7.1                 |
+-----------+--------------------------+

基础环境准备请参照这份 `文档 <https://gitcode.com/Ascend/pytorch>`_ 。

### 安装VeOmni

.. code-block:: bash

    git clone https://github.com/ByteDance-Seed/VeOmni.git
    cd VeOmni
    pip install -e .[npu]

### 快速开始

1. 准备模型和数据集

2. 设置`train.sh`中`NPROC_PER_NODE`参数为实际卡数

3. 运行训练脚本

.. code-block:: bash
    # 设置环境变量
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True 
    export MULTI_STREAM_MEMORY_REUSE=2 
    bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml

### 并行能力支持

+-----------+--------------------------+
| 能力       | 是否支持                  |
+-----------+--------------------------+
| fsdp      | ✅                       |
+-----------+--------------------------+
| fsdp2     | ✅                       |
+-----------+--------------------------+
| ulysses parallel    | ✅             |
+-----------+--------------------------+
| expert_parallel | 适配中              |
+-----------+--------------------------+

