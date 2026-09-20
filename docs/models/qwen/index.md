# Qwen

Text, vision-language, audio/video, and image generation across the Qwen model families.

## Models and workflows

| Model | Checkpoints in these recipes | Workflow |
| --- | --- | --- |
| [Qwen3](qwen3.md) | 0.6B / 8B; 30B MoE example | Dense text SFT and LoRA |
| [Qwen3 MoE](qwen3-moe.md) | 30B-A3B | Expert-parallel training, LoRA, and Muon |
| [Qwen3.5](qwen3-5.md) | 0.8B / 9B / 35B-A3B | Dense and MoE text / vision-language training |
| [Qwen3.8 Flash Next](qwen3-8-flash-next.md) | Qwen3.8-Flash-Next | Qwen4-Exp with PLE and expert parallelism |
| [Qwen3 VL](qwen3-vl.md) | 8B / 30B-A3B | Dense and MoE image/video SFT |
| [Qwen3 Omni](qwen3-omni.md) | 30B-A3B thinker | Multisource thinker training |
| [Qwen3 Omni: offline AV](qwen3-omni-offline-av.md) | 30B-A3B thinker | Pre-extracted audio-enabled video |
| [Qwen3 DPO](qwen3-dpo.md) | 0.6B | Preference optimization |
| [Qwen2.5](qwen2-5.md) | 7B-Instruct | Plain-text training |
| [Qwen2 / Qwen2.5 VL](qwen2-vl.md) | 7B-Instruct | Image and video training |
| [Qwen2.5 Omni](qwen2-5-omni.md) | 7B | Text, image, video, and audio inputs |
| [Qwen-Image](qwen-image.md) | Qwen-Image | Diffusion SFT and LoRA |

## Choose a workflow

- First training run: use the validated [Qwen3-0.6B Quick Start](../../get_started/quick_start.md).
- Text SFT or adapters: start with Qwen3; use the dedicated MoE recipe for expert parallelism.
- Images and video: choose Qwen3 VL or Qwen3.5 and prepare media-aware conversations.
- Audio together with video: use Qwen3 Omni; offline AV is a separate data preparation path.
- Preference pairs: use Qwen3 DPO. Text-to-image diffusion: use Qwen-Image.

## Before launching

Each model page lists its checkpoint, data format, launch command, and configuration
choices. Read the [validation scope](../index.md#choose-hardware-and-check-a-run)
and [hardware guide](../../hardware_support/index.md) before scaling to a full model.

```{toctree}
:hidden:
:maxdepth: 1

Qwen3 <qwen3>
Qwen3 MoE <qwen3-moe>
Qwen3.5 <qwen3-5>
Qwen3.8 Flash Next <qwen3-8-flash-next>
Qwen3 VL <qwen3-vl>
Qwen3 Omni <qwen3-omni>
Qwen3 Omni: offline AV <qwen3-omni-offline-av>
Qwen3 DPO <qwen3-dpo>
Qwen2.5 <qwen2-5>
Qwen2 / Qwen2.5 VL <qwen2-vl>
Qwen2.5 Omni <qwen2-5-omni>
Qwen-Image <qwen-image>
```
