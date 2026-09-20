# Wan

Video diffusion training with full-model and adapter workflows. Select the checkpoint for the conditioning task.

## Models and workflows

| Model | Workflow |
| --- | --- |
| [Wan2.1 I2V](wan2-1-i2v.md) | 14B image-to-video training |
| [Wan2.1 T2V LoRA](wan2-1-t2v.md) | 1.3B text-to-video; online/offline preprocessing |

## Before launching

Each model page lists its checkpoint, data format, launch command, and configuration
choices. Read the [validation scope](../index.md#choose-hardware-and-check-a-run)
and [hardware guide](../../hardware_support/index.md) before scaling to a full model.

```{toctree}
:hidden:
:maxdepth: 1

Wan2.1 I2V <wan2-1-i2v>
Wan2.1 T2V LoRA <wan2-1-t2v>
```
