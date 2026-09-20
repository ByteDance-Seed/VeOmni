# DeepSeek

Choose the recipe for the model generation: V4 uses a different attention and kernel stack from V2/V3/R1.

## Models and workflows

| Model | Workflow |
| --- | --- |
| [DeepSeek V2 / V3 / R1](deepseek-v3.md) | Plain-text training and V3 LoRA |
| [DeepSeek V4](deepseek-v4.md) | GPU / NPU recipes and indexer training |

## Before launching

Each model page lists its checkpoint, data format, launch command, and configuration
choices. Read the [validation scope](../index.md#choose-hardware-and-check-a-run)
and [hardware guide](../../hardware_support/index.md) before scaling to a full model.

```{toctree}
:hidden:
:maxdepth: 1

DeepSeek V2 / V3 / R1 <deepseek-v3>
DeepSeek V4 <deepseek-v4>
```
