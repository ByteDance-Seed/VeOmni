# Welcome to VeOmni

VeOmni is a PyTorch-native framework for text, vision-language, audio/video, and
diffusion model training. Reusable model runtimes, data pipelines, and distributed
components support pre-training and post-training workflows.

## Start here

| Your goal | Guide |
| --- | --- |
| Run your first training job | [Quick Start](get_started/quick_start.md) |
| Install for your hardware | [Getting Started](get_started/index.md) |
| Choose a model and configuration | [Models and Recipes](examples/index.md) |
| Prepare data or manage checkpoints | [User Guide](usage/index.md) |
| Configure parallelism or LoRA | [Features](key_features/index.md) |
| Look up a configuration field | [Reference](reference/index.md) |
| Extend or contribute to VeOmni | [Developer Guide](developer/index.md) |

The `latest` documentation tracks `main`. For an older checkout, use that
revision's documentation and configurations together. See the
[project README](https://github.com/ByteDance-Seed/VeOmni) for milestones,
roadmaps, and community links.

```{toctree}
:hidden:
:maxdepth: 2

get_started/index
usage/index
examples/index
key_features/index
reference/index
developer/index
design/index
```

## Citation

If you find VeOmni useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@article{ma2025veomni,
  title={VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo},
  author={Ma, Qianli and Zheng, Yaowei and Shi, Zhelun and Zhao, Zhongkai and Jia, Bin and Huang, Ziyue and Lin, Zhiqi and Li, Youjie and Yang, Jiacheng and Peng, Yanghua and others},
  journal={arXiv preprint arXiv:2508.02317},
  year={2025}
}
```
