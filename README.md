
<div align="center">

<img src="./assets/logo.png" width="50%">

## VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo

<p align="center">
  <a href="https://github.com/ByteDance-Seed/VeOmni/stargazers">
    <img src="https://img.shields.io/github/stars/ByteDance-Seed/VeOmni?style=social"></a>
  <a href="https://github.com/ByteDance-Seed/VeOmni">
    <img src="https://img.shields.io/badge/VeOmni-Project Page-yellow"></a>
  <a href="https://arxiv.org/abs/2508.02317">
    <img src="https://img.shields.io/badge/VeOmni-Tech Report-red"></a>
  <a href="https://huggingface.co/ByteDance-Seed">
    <img src="https://img.shields.io/badge/VeOmni-Hugging Face-orange"></a>
  <br>
  <a href="https://github.com/ByteDance-Seed/VeOmni/asserts/wechat.png">
    <img src="https://img.shields.io/badge/VeOmni-Wechat Communication Group-07C160"></a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-blue"></a>
</p>

</div>

## 🔗 Overview
VeOmni is a versatile framework for both single- and multi-modal pre-training and post-training. It empowers users to seamlessly scale models of any modality across various accelerators, offering both flexibility and user-friendliness.

Our guiding principles when building VeOmni are:
- **Flexibility and Modularity**: VeOmni is built with a modular design, allowing users to decouple most components and replace them with their own implementations as needed.
- **Trainer-free**: VeOmni avoids rigid, structured trainer classes (e.g., [PyTorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning) or [HuggingFace](https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/trainer#transformers.Trainer) Trainer). Instead, VeOmni keeps training scripts linear, exposing the entire training logic to users for maximum transparency and control.

- **Omni model native**: VeOmni enables users to effortlessly scale any omni-model across devices and accelerators.
- **Torch native**: VeOmni is designed to leverage PyTorch’s native functions to the fullest extent, ensuring maximum compatibility and performance.

<div align="center">
<img src="./assets/system.png" width="100%">
</div>

### 🔥 Latest News
- [2025/1/11] Our Paper [OmniScale: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo]() was accepted by AAAI 2026
- [2025/09/19] We release first offical release [v0.1.0](https://github.com/ByteDance-Seed/VeOmni/pull/75) of VeOmni.
- [2025/08/01] We release [VeOmni Tech report](https://arxiv.org/abs/2508.02317) and open the [WeChat group](./assets/wechat.png). Feel free to join us!
- [2025/04/03] We release VeOmni!


## 🔖 Table of Contents

- [VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo](#veomni-scaling-any-modality-model-training-with-model-centric-distributed-recipe-zoo)
- [🔗 Overview](#-overview)
  - [🔥 Latest News](#-latest-news)
- [🔖 Table of Contents](#-table-of-contents)
- [📚 Key Features](#-key-features)
  - [🧪 Upcoming Features](#-upcoming-features)
- [🎈 Getting Started](#-getting-started)
  - [🔧 Installation](#-installation)
    - [(Recommended) Use `uv` Managed Virtual Environment](#recommended-use-uv-managed-virtual-environment)
    - [`pip` Based Install](#pip-based-install)
  - [🚀 Quick Start](#-quick-start)
  - [🔒 Merge checkpoints](#-merge-checkpoints)
  - [📦 Build Docker](#-build-docker)
- [🧱 Training Examples](#-training-examples)
- [✏️ Supported Models](#️-supported-models)
- [⛰️ Performance](#️-performance)
- [😊 Acknowledgement](#-acknowledgement)
- [💡 Awesome work using VeOmni](#-awesome-work-using-veomni)
- [🎨 Contributing](#-contributing)
- [📄 License](#-license)
- [📝 Citation](#-citation)
- [🌱 About ByteDance Seed Team](#-about-bytedance-seed-team)

## 📚 Key Features

- **Parallelism**
  - Parallel state by [DeviceMesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
  - Torch FSDP1/2
  - Experts parallelism(Experimental)
  - Easy to add new parallelism plan
  - Sequence parallelism
    - [Ulysess](https://arxiv.org/abs/2309.14509)
    - Async-Ulysses
  - Activation offloading
  - Activation checkpointing
- **Kernels**
  - GroupGemm ops for moe
  - [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) integrations
- **Model**
  - Any [transformers](https://github.com/huggingface/transformers) models.
  - Multi-modal
    - Qwen2.5-VL
    - Qwen2-VL
    - Seed-Omni
- **Data IO**
  - Dynamic batching strategy
  - Omnidata processing
- **Distributed Checkpointing**
  - [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint)
  - Torch Distributed checkpointing
  - Dcp merge tools
- **Other tools**
  - Profiling tools
  - Easy yaml configuration and argument parsing

### 🧪 Upcoming Features

- [ ] Torch native Tensor parallelism
- [ ] torch.compile
- [ ] [Flux: Fine-grained Computation-communication Overlapping GPU Kernel](https://github.com/bytedance/flux) integrations
- [ ] Better offloading strategy
- [ ] More models support
- [ ] Torch native pipeline parallelism


## 🎈 Getting Started

Read the [VeOmni Best Practice](docs/start/best_practice.md) for more details.

### 🔧 Installation

#### (Recommended) Use `uv` Managed Virtual Environment

We recommend to use [`uv`](https://docs.astral.sh/uv/) managed virtual environment
to run VeOmni.

```shell
# For GPU
uv sync --extra gpu
# For Ascend NPU
uv sync --extra npu
# You can install other optional deps by adding --extra like --extra dit

# Activate the uv managed virtual environment
source .venv/bin/activate
```

#### `pip` Based Install

Install using PyPI:

```shell
pip3 install veomni
```

Install from source code:

```shell
pip3 install -e .
```

### 🚀 Quick Start

User can quickly start training like this:

```shell
bash train.sh $TRAIN_SCRIPT $CONFIG.yaml
```

You can also override arguments in yaml by passing arguments from an external command line:

```shell
bash train.sh $TRAIN_SCRIPT $CONFIG.yaml \
    --model.model_path PATH/TO/MODEL \
    --data.train_path PATH/TO/DATA \
    --train.global_batch_size GLOBAL_BATCH_SIZE \
```

Here is an end-to-end workflow for preparing a subset of the fineweb dataset, continuing training a qwen2_5 model with sequence parallel 2 for 20 steps, and then merging the global_step_10 distributed checkpoint to hf weight by ByteCheckpoint.

1. Download fineweb dataset

```shell
python3 scripts/download_hf_data.py \
  --repo_id HuggingFaceFW/fineweb \
  --local_dir ./fineweb/ \
  --allow_patterns sample/10BT/*
```

2. Download qwen2_5 model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen2.5-7B \
  --local_dir .
```

3. Training

```shell
bash train.sh tasks/train_torch.py configs/pretrain/qwen2_5.yaml \
    --model.model_path ./Qwen2.5-7B \
    --data.train_path ./fineweb/sample/10BT/ \
    --train.global_batch_size 512 \
    --train.lr 5e-7 \
    --train.ulysses_parallel_size 2 \
    --train.save_steps 10 \
    --train.max_steps 20 \
    --train.output_dir Qwen2.5-7B_CT
```

4. Merge checkpoints

```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10 \
    --model_assets_dir Qwen2.5-7B-Instruct_CT/model_assets \
    --save-dir Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10/hf_ckpt
```

5. Inference

```shell
python3 tasks/infer.py \
  --infer.model_path Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10/hf_ckpt
```


### 🔒 Merge checkpoints

we use [ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint) to save checkpoints in torch.distributed.checkpoint(dcp) format. You can merge the dcp files using this command:

```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir PATH/TO/CHECKPOINTS \
    --model_assets_dir PATH/TO/MODEL_ASSETS \
    --save-dir PATH/TO/SAVE_HF_WEIGHT \
```

For example, your output_dir is `seed_omni`, and you want to merge global_step_100 checkpoint to huggingface-type weight:

```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir seed_omni/checkpoints/global_step_100 \
    --model_assets_dir seed_omni/model_assets \
    --save-dir seed_omni/hf_ckpt \
```

### 📦 Build Docker

```shell
cd docker/
docker compose up -d
docker compose exec VeOmni bash
```

## 🧱 Training Examples

PyTorch FSDP2 Qwen2VL

```shell
bash train.sh tasks/multimodal/omni/train_qwen2_vl.py configs/multimodal/qwen2_vl/qwen2_vl.yaml
```

PyTorch FSDP2 Qwen2

```shell
bash train.sh tasks/train_torch.py configs/pretrain/qwen2_5.yaml
```

PyTorch FSDP2 llama3-8b-instruct

```shell
bash train.sh  tasks/train_torch.py configs/pretrain/llama3.yaml
```

## ✏️ Supported Models

| Model                                                    | Model size                    | Example config File                                                   |
| -------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------|
| [DeepSeek 2.5/3/R1](https://huggingface.co/deepseek-ai)  | 236B/671B                     | [deepseek.yaml](configs/pretrain/deepseek.yaml)                       |
| [Llama 3-3.3](https://huggingface.co/meta-llama)         | 1B/3B/8B/70B                  | [llama3.yaml](configs/pretrain/llama3.yaml)                           |
| [Qwen 2-3](https://huggingface.co/Qwen)                  | 0.5B/1.5B/3B/7B/14B/32B/72B/  | [qwen2_5.yaml](configs/pretrain/qwen2_5.yaml)                         |
| [Qwen2-3 VL/QVQ](https://huggingface.co/Qwen)            | 2B/3B/7B/32B/72B              | [qwen3_vl_dense.yaml](configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml)|
| [Qwen3-VL MoE](https://huggingface.co/Qwen)              | 30BA3B/235BA22B               | [qwen3_vl_moe.yaml](configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml)    |
| [Qwen3-MoE](https://huggingface.co/Qwen)                 | 30BA3B/235BA22B               | [qwen3-moe.yaml](configs/pretrain/qwen3-moe.yaml)                     |
| [Wan](https://huggingface.co/Wan-AI)                     | Wan2.1-I2V-14B-480P           | [wan_sft.yaml](configs/dit/wan_sft.yaml)                              |
| Omni Model                                               | Any Modality Training         | [seed_omni.yaml](configs/multimodal/omni/seed_omni.yaml)              |


> VeOmni Support all [transformers](https://github.com/huggingface/transformers) models if you don't need sequence parallelism or experts parallelism or other parallelism and cuda kernal optimize in VeOmni. We design a [model registry mechanism](veomni/models/registry.py). When the model is registered in veomni, we will automatically load the model and optimizer in VeOmni. Otherwise, it will default to load the modeling file in transformers.

> If you want to add a new model, you can add a new model in the model registry. See in [Support costom model](docs/tutorials/model_loader.md) docs.

## ⛰️ Performance

Seed in Tech report (https://arxiv.org/abs/2508.02317)

## 😊 Acknowledgement

Thanks to the following projects for their excellent work:

- [ByteCheckpoint](https://arxiv.org/abs/2407.20143)
- [veScale](https://github.com/volcengine/veScale)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [torchtitan](https://github.com/pytorch/torchtitan/)
- [torchtune](https://github.com/pytorch/torchtune)

## 💡 Awesome work using VeOmni
- [dFactory: Easy and Efficient dLLM Fine-Tuning](https://github.com/inclusionAI/dFactory)
- [LMMs-Engine](https://github.com/EvolvingLMMs-Lab/lmms-engine)
- [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://github.com/bytedance/UI-TARS)
- [OpenHA: A Series of Open-Source Hierarchical
Agentic Models in Minecraft](https://arxiv.org/pdf/2509.13347)
- [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.02544)
- [Open-dLLM: Open Diffusion Large Language Models](https://github.com/pengzhangzhi/Open-dLLM)

## 🎨 Contributing

Contributions from the community are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) our project roadmap(To be updated),

## 📄 License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.


## 📝 Citation

If you find VeOmni useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@article{ma2025veomni,
  title={VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo},
  author={Ma, Qianli and Zheng, Yaowei and Shi, Zhelun and Zhao, Zhongkai and Jia, Bin and Huang, Ziyue and Lin, Zhiqi and Li, Youjie and Yang, Jiacheng and Peng, Yanghua and others},
  journal={arXiv preprint arXiv:2508.02317},
  year={2025}
}
```

## 🌱 About [ByteDance Seed Team](https://team.doubao.com/)

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

You can get to know us better through the following channels👇
<p align="center">
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>
