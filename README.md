<div align="center">

<img src="./docs/assets/logo.png" width="50%">

# VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo

[![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/VeOmni)](https://github.com/ByteDance-Seed/VeOmni/stargazers)
[![Paper](https://img.shields.io/badge/Paper-red)](https://arxiv.org/abs/2508.02317)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://veomni.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

</div>

## Overview

VeOmni is a versatile distributed training framework for both single- and multi-modal pre-training and post-training. It empowers users to seamlessly scale models of any modality — text, vision-language, omni-modal, and diffusion — across GPUs and NPUs, offering both flexibility and production-grade performance.

Our guiding principles when building VeOmni are:

- **Flexibility and Modularity**: VeOmni is built with a modular design, allowing users to decouple most components and replace them with their own implementations as needed.
- **Trainer-free**: VeOmni supports linear training scripts that avoid rigid, structured trainer classes. These scripts expose the entire training logic for maximum transparency and control. For those who prefer them, a basic trainer for text-only or VLM/omni models and an RL trainer backend are also provided.
- **Omni model native**: VeOmni enables users to effortlessly scale any omni-model across devices and accelerators.
- **Torch native**: VeOmni is designed to leverage PyTorch's native functions to the fullest extent, ensuring maximum compatibility and performance.

## Key Capabilities

- **Broad model coverage** — Ready-to-run recipes for DeepSeek-V3/R1/V4, Llama 3, Qwen2/3 families (dense, MoE, VL, Omni), GPT-OSS, Wan, LTX-2.3, and Kimi K3 (multimodal MoE, Ascend NPU).
- **Model-centric parallel recipes** — Data parallel (FSDP2), tensor parallel, pipeline parallel, expert parallel, sequence parallel (Ulysses), and context parallel can be composed per model through declarative config, without touching training code.
- **Patch-generation model adaptation** — Each model lives as a small, reviewable patch-generation config instead of a forked modeling file. Patches are regenerated from upstream sources, keeping model code in sync with Hugging Face Transformers and making drift detectable.
- **Per-backend kernel registry** — The ops registry selects optimized kernels per backend automatically (CUDA, AscendC NPU, TileLang, Triton), with eager fallbacks where a fused kernel is unavailable.
- **Training features** — Mixed precision, gradient checkpointing (including selective recompute under expert parallel), dynamic token-budget batching, LoRA, distributed checkpointing (DCP) with resharding across parallel layouts, and built-in profiling.

## News

- [2026/10] Kimi K3 multimodal MoE support: AscendC NPU kernels with a verified end-to-end SFT recipe and training guide.
- [2025/11] Our paper [OmniScale: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo](https://arxiv.org/abs/2508.02317) was accepted by AAAI 2026.
- [2025/09] First official release [v0.1.0](https://github.com/ByteDance-Seed/VeOmni/pull/75) of VeOmni.
- [2025/04] We release VeOmni!

## Getting Started

### Installation

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
pip install -e .
```

For environment-specific instructions (CUDA extensions, NPU toolchains, Docker images), see the [Installation Guide](https://veomni.readthedocs.io/en/latest/get_started/installation/install.html). Prebuilt images for GPU and NPU environments are available under [`docker/`](./docker).

### Quick Start

A training run is one command — pick a task entry point and a recipe config:

```bash
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml
```

Entry points for other tasks (text, diffusion, RL) live under [`tasks/`](./tasks). For a fully worked example, see [Quick Start with Qwen3](https://veomni.readthedocs.io/en/latest/examples/qwen3.html).

## Repository Layout

```
VeOmni/
├── veomni/                  # Core framework package
│   ├── arguments/           #   Dataclass-based CLI/YAML config schema
│   ├── checkpoint/          #   Distributed checkpoint save/load/resharding
│   ├── data/                #   Packing, dynamic batching, multimodal transforms
│   ├── distributed/         #   Process-group & parallelism initialization
│   ├── lora/                #   LoRA adapters
│   ├── models/              #   Per-model registries and patch-generation configs
│   ├── ops/                 #   Kernel registry with per-backend selection
│   ├── optim/               #   Optimizers
│   ├── patchgen/            #   Patch-generation engine
│   ├── schedulers/          #   LR schedulers
│   └── trainer/             #   Trainer backends (SFT, RL)
├── tasks/                   # Training entry points (train_vlm.py, ...)
├── configs/                 # Ready-to-run YAML recipes (text / multimodal / dit)
│   └── multimodal/data/     #   Dataset descriptor YAMLs
├── scripts/                 # Data conversion and utility scripts
│   └── multimodal/convert_data/
├── tests/                   # Toy model configs and CI
│   └── toy_config/          #   Reduced-scale model configs for fast validation
├── docs/                    # Documentation and per-model training guides
│   └── examples/            #   e.g. kimi_k3.md, qwen3_vl.md, wan2.1.md
├── docker/                  # Dockerfiles for GPU/NPU environments
├── patchgen-pkg/            # Standalone patch-generation tooling
└── .agents/skills/          # Agent skills: develop/debug/profile/patchgen workflows
```

## Usage Guide

### Run a recipe

```bash
bash train.sh <tasks/entry_point.py> <configs/recipe.yaml> [overrides]
```

A recipe config has three logical sections: `model` (architecture or checkpoint path), `data` (dataset descriptors and packing budget), and `train` (parallel layout, batch schedule, checkpointing, profiling). Any field can be overridden on the command line, e.g. `train.profile.enable true`. See [Usage Documentation](https://veomni.readthedocs.io/en/latest/) for the full reference.

### Verify a setup quickly

Reduced-scale model configs under [`tests/toy_config/`](./tests/toy_config) let you validate a full training pipeline (data, parallelism, kernels, checkpointing) in minutes at toy cost. Most models ship a paired smoke recipe — for example, the Kimi K3 end-to-end recipe:

```bash
# prepare a small caption dataset (COCA-style) and a HF-format tokenizer, then:
bash train.sh tasks/train_vlm.py configs/multimodal/kimi_k3/kimi_k3_vl_ascendc.yaml
```

The full guide (data preparation, model contracts, verification checklist) is in [`docs/examples/kimi_k3.md`](./docs/examples/kimi_k3.md). Guides for other models live alongside it in [`docs/examples/`](./docs/examples).

### Add a new model

VeOmni adapts models through the patch-generation pipeline rather than vendored forks: describe your modifications in a patch-generation config, generate the patched modeling file, and register per-backend kernels through the ops registry. The step-by-step guide is [Support New Models](https://veomni.readthedocs.io/en/latest/usage/support_new_models/guide_and_checklist.html), and [`.agents/skills/veomni-patchgen-model`](./.agents/skills) encodes the complete workflow for both humans and AI coding agents.

### Profile and debug

Set `train.profile.enable true` (plus `start_step` / `end_step`) on any run to export step timelines, per-op statistics, and communication breakdowns — no code changes needed. For systematic debugging workflows, see [`.agents/skills/veomni-debug`](./.agents/skills) and [`.agents/skills/veomni-profile`](./.agents/skills).

### Backends

The same recipe runs on GPU and NPU clusters: the ops registry resolves each kernel to the best available backend implementation (CUDA/flash-attn, AscendC, TileLang, Triton) with automatic eager fallback. Backend-specific recipes are suffixed accordingly (e.g. `*_ascendc.yaml`).

## Supported Models

| Model | Model size | Example config |
| --- | --- | --- |
| [DeepSeek2.5/3/R1](https://huggingface.co/deepseek-ai) | 236B/671B | [deepseek.yaml](configs/text/deepseek.yaml) |
| [DeepSeek-V4](https://huggingface.co/deepseek-ai) | Checkpoint-dependent | [deepseek.yaml](configs/text/deepseek.yaml); DSA/mHC default to eager, with optional [SM90+ TileLang/TileKernels backends](docs/design/op_selection.md#deepseek-v4-dsa-and-mhc) |
| [Llama3-3.3](https://huggingface.co/meta-llama) | 1B/3B/8B/70B | [llama3.yaml](configs/text/llama3.yaml) |
| [Qwen2-3](https://huggingface.co/Qwen) | 0.5B-72B | [qwen2_5.yaml](configs/text/qwen2_5.yaml) |
| [Qwen2-3 VL/QVQ](https://huggingface.co/Qwen) | 2B-72B | [qwen3_vl_dense.yaml](configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml) |
| [Qwen3-VL MoE](https://huggingface.co/Qwen) | 30BA3B/235BA22B | [qwen3_vl_moe.yaml](configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml) |
| [Qwen3-MoE](https://huggingface.co/Qwen) | 30BA3B/235BA22B | [qwen3-moe.yaml](configs/text/qwen3-moe.yaml) |
| [GPT-OSS](https://huggingface.co/openai/gpt-oss-120b) | 120B | [gpt_oss_120b_lora_ep4.yaml](configs/text/gpt_oss_120b_lora_ep4.yaml) |
| [Qwen2-3 Omni](https://huggingface.co/Qwen) | 7B/30BA3B | [qwen25_omni.yaml](configs/multimodal/qwen25_omni/qwen25_omni.yaml) |
| Kimi K3 (multimodal MoE) | Checkpoint-dependent | [kimi_k3_vl_ascendc.yaml](configs/multimodal/kimi_k3/kimi_k3_vl_ascendc.yaml); [training guide](docs/examples/kimi_k3.md) |
| [Wan](https://huggingface.co/Wan-AI) | Wan2.1-I2V-14B-480P | [wan_sft.yaml](configs/dit/wan_sft.yaml) |
| [MiniMax H3](docs/examples/minimax_h3.md) | Checkpoint-dependent | [FL2VA](configs/dit/minimax_h3_fl2va_offline.yaml); [Ref2VA](configs/dit/minimax_h3_ref2va_cfg_offline.yaml) |
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | Checkpoint-dependent | [ltx2_av_lora.yaml](configs/dit/ltx2_av_lora.yaml); [training guide](docs/examples/ltx-2.3.md) |

To support a new model, see [Support New Models](https://veomni.readthedocs.io/en/latest/usage/support_new_models/guide_and_checklist.html).

## Performance

Benchmark results and throughput comparisons are maintained in the [documentation](https://veomni.readthedocs.io/en/latest/). Reproduction scripts accompany each recipe config.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

- **Code style** — The codebase is formatted and linted with `ruff`; run it over your changes before submitting.
- **Model code** — Changes to patched model files must go through the patch-generation pipeline (never hand-edit generated files); regenerate and commit patches together with the patch-generation config. The workflow is encoded in [`.agents/skills/veomni-patchgen-model`](./.agents/skills).
- **Validation** — Include a reduced-scale run (see `tests/toy_config/`) for any change that touches data, parallelism, or kernels.
- **Agent skills** — AI coding agents working on this repository should follow [`.agents/skills/README.md`](./.agents/skills/README.md); the skills cover development, debugging, profiling, and model-adaptation workflows.

## Citation and Acknowledgement

If you find VeOmni useful for your research and applications, feel free to give us a star or cite us using:

```bibtex
@article{ma2025veomni,
  title={VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo},
  author={Ma, Qianli and Zheng, Yaowei and Shi, Zhelun and Zhao, Zhongkai and Jia, Bin and Huang, Ziyue and Lin, Zhiqi and Li, Youjie and Yang, Jiacheng and Peng, Yanghua and others},
  journal={arXiv preprint arXiv:2508.02317},
  year={2025}
}
```

Thanks to the following projects for their excellent work:

- [ByteCheckpoint](https://arxiv.org/abs/2407.20143)
- [veScale](https://github.com/volcengine/veScale)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [torchtitan](https://github.com/pytorch/torchtitan/)
- [torchtune](https://github.com/pytorch/torchtune/)

## License

This project is released under the [Apache 2.0 License](./LICENSE).

## About

VeOmni is developed by the [ByteDance Seed Team](https://team.doubao.com/). Join our WeChat group via the QR code in the [documentation](https://veomni.readthedocs.io/en/latest/) to stay in touch with the community.
