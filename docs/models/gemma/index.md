# Gemma

Text-model training; condition encoders used by diffusion models are documented in their own recipes.

## Models and workflows

| Model | Workflow |
| --- | --- |
| [Gemma3](gemma3.md) | 270M text training with FlexAttention |

## Before launching

Each model page lists its checkpoint, data format, launch command, and configuration
choices. Read the [validation scope](../index.md#choose-hardware-and-check-a-run)
and [hardware guide](../../hardware_support/index.md) before scaling to a full model.

```{toctree}
:hidden:
:maxdepth: 1

Gemma3 <gemma3>
```
