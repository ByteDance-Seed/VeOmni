# Welcome to VeOmni

[![GitHub Repo stars](https://img.shields.io/github/stars/ByteDance-Seed/VeOmni)](https://github.com/ByteDance-Seed/VeOmni/stargazers)
[![Paper](https://img.shields.io/badge/Paper-red)](https://arxiv.org/abs/2508.02317)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://veomni.readthedocs.io/en/latest/)
[![WeChat](https://img.shields.io/badge/WeChat-green?logo=wechat&amp)](https://raw.githubusercontent.com/ByteDance-Seed/VeOmni/refs/heads/main/docs/assets/wechat.png)


VeOmni is a versatile framework for both single- and multi-modal pre-training and post-training. It empowers users to seamlessly scale models of any modality across various accelerators, offering both flexibility and user-friendliness.


---

```{toctree}
:maxdepth: 1
:caption: Get Started

get_started/installation/install.md
get_started/installation/install_ascend.md
```

```{toctree}
:maxdepth: 1
:caption: Usage

usage/basic_modules.md
usage/arguments.md
usage/support_new_models.md
usage/checkpoint_conversion.md
```

```{toctree}
:maxdepth: 1
:caption: Hardware Support

hardware_support/get_started_npu.md
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/qwen3.md
examples/qwen3_vl.md
examples/wan2.1.md
```

```{toctree}
:maxdepth: 1
:caption: Key Features

key_features/ep_fsdp2.md
key_features/ulysses.md
key_features/preprocessor_registry.md
```

```{toctree}
:maxdepth: 1
:caption: Transformers v5 Updates

transformers_v5/index.md
```

---

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

## About [ByteDance Seed Team](https://team.doubao.com/)

<div align="center">
<img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" width="100%">
</div>

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society. You can get to know Bytedance Seed better through the following channels👇
<div>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>

</div>
