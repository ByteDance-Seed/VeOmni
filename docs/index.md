# VeOmni

<p class="doc-lead">Train language, vision, audio, and diffusion models with a composable PyTorch framework.</p>

VeOmni combines reusable model runtimes, data pipelines, and distributed training
components for pre-training and post-training. Start with a short training run,
or go directly to the recipe for your model.

<div class="doc-cards">
<a class="doc-card" href="get_started/quick_start.html"><strong>Quick Start →</strong><span>Prepare a small dataset, train Qwen3, and verify your first checkpoint.</span></a>
<a class="doc-card" href="models/index.html"><strong>Supported Models →</strong><span>Explore model families and complete text, multimodal, and diffusion recipes.</span></a>
<a class="doc-card" href="hardware_support/index.html"><strong>Installation & hardware →</strong><span>Choose the environment and kernel settings for your accelerator.</span></a>
<a class="doc-card" href="developer/index.html"><strong>Developer Guide →</strong><span>Understand the architecture, integrate a model, and contribute changes.</span></a>
</div>

## Training workflows

- **Language models:** plaintext pre-training, supervised fine-tuning, LoRA, and DPO.
- **Multimodal models:** image, video, and audio conversations with model-specific processors.
- **Diffusion models:** full-model training, adapters, and offline conditioning pipelines.
- **Distributed execution:** FSDP2, Ulysses sequence parallelism, and expert parallelism.

See [Models](models/index.md) for the recipe-specific combinations and
[Features](key_features/index.md) for configuration and usage.

## Explore the documentation

| Section | What you will find |
| --- | --- |
| [User Guide](guide/index.md) | Installation, data, checkpoints, and configuration reference |
| [Models](models/index.md) | Family overviews and model-specific preparation / launch recipes |
| [Features](key_features/index.md) | Parallelism, LoRA, and training features |
| [Developer Guide](developer/index.md) | Architecture, model integration, and contribution workflows |
| [Design](design/index.md) | Implementation contracts and migration history |
| [Hardware](hardware_support/index.md) | Platform setup, evidence, and operating notes |

The `latest` documentation follows `main`. For another checkout, use documentation
and configurations from the same revision. See the
[project README](https://github.com/ByteDance-Seed/VeOmni) for milestones and community links.

```{toctree}
:hidden:
:maxdepth: 3

User Guide <guide/index>
Models <models/index>
Features <key_features/index>
Developer Guide <developer/index>
Design <design/index>
Hardware <hardware_support/index>
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
