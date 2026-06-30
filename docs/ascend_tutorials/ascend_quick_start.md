# Ascend Quickstart

Last updated: 11/28/2025

我们在VeOmni上增加对华为昇腾设备的支持。

### 基础环境准备

| software  | version                  |
|-----------|--------------------------|
| Python    | >= 3.10, <3.12           |
| CANN      | == 8.3.RC1               |
| torch     | == 2.7.1                 |
| torch_npu | == 2.7.1                 |

基础环境准备请参照这份 [文档](https://gitcode.com/Ascend/pytorch) 。

### 使用uv安装依赖环境

#### 1. 进入VeOmni仓的根目录

    git clone https://github.com/ByteDance-Seed/VeOmni.git
    cd VeOmni

#### 2. 固定python版本

    uv python pin 3.11

#### 3. 设置超时时间（可选）

如果网络不稳定，可以设置超时时间防止下载超时。设置环境变量 UV_HTTP_TIMEOUT 来调整超时时间：

    export UV_HTTP_TIMEOUT=60

#### 4. 执行 uv 环境安装

    uv sync --extra npu --allow-insecure-host github.com --allow-insecure-host pythonhosted.org

#### 5. 环境使用

安装好之后，在VeOmni根目录会出现一个.venv文件夹，就是uv安装的环境。可用如下命令激活环境：

    source .venv/bin/activate

查看安装的依赖包列表：

    uv pip list 

### 快速开始

1. 准备模型和数据集

2. 设置`train.sh`中`NPROC_PER_NODE`参数为实际卡数

3. 运行训练脚本

```bash
# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2

bash train.sh tasks/train_torch.py configs/sft/qwen3_sft.yaml
```

### 并行能力支持

| 能力               | 是否支持               |
|-------------------|----------------------|
| fsdp              | ✅                   |
| fsdp2             | ✅                   |
| ulysses parallel  | ✅                   |
| expert_parallel   | 适配中                |

