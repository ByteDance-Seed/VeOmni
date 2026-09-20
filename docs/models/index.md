# Supported Models

<p class="doc-lead">Explore VeOmni recipes by model family — from language and multimodal training to image and video generation.</p>

Choose a family below to compare workflows, or open a model directly for its
preparation, launch command, and configuration.

## Language and multimodal models

````{container} model-grid

```{container} model-family featured

**[Qwen](qwen/index.md)**

Text · vision-language · audio/video · preference training

- [Qwen3-0.6B / 8B](qwen/qwen3.md)
- [Qwen3-30B-A3B](qwen/qwen3-moe.md)
- [Qwen3.5-0.8B / 9B / 35B-A3B](qwen/qwen3-5.md)
- [Qwen3.8 Flash Next](qwen/qwen3-8-flash-next.md)
- [Qwen3-VL-8B / 30B-A3B](qwen/qwen3-vl.md)
- [Qwen3 Omni](qwen/qwen3-omni.md)
- [Qwen3 Omni: offline AV](qwen/qwen3-omni-offline-av.md)
- [Qwen3 DPO](qwen/qwen3-dpo.md)
- [Qwen2.5](qwen/qwen2-5.md)
- [Qwen2 / Qwen2.5 VL](qwen/qwen2-vl.md)
- [Qwen2.5 Omni](qwen/qwen2-5-omni.md)

```

```{container} model-family

**[DeepSeek](deepseek/index.md)**

MoE training · generation-specific kernels

- [DeepSeek V2 / V3 / R1](deepseek/deepseek-v3.md)
- [DeepSeek V4](deepseek/deepseek-v4.md)

```

```{container} model-family

**[Llama](llama/index.md)**

Plain-text continued pre-training

- [Llama3](llama/llama3.md)

```

```{container} model-family

**[Gemma](gemma/index.md)**

Text training with FlexAttention

- [Gemma3-270M](gemma/gemma3.md)

```

```{container} model-family

**[GPT-OSS](gpt-oss/index.md)**

BF16 adapter training with expert parallelism

- [GPT-OSS-120B LoRA](gpt-oss/gpt-oss.md)

```

```{container} model-family

**[Seed](seed/index.md)**

Seed-OSS language-model training

- [Seed-OSS-36B](seed/seed-oss.md)

```

````

## Diffusion models

````{container} model-grid

```{container} model-family

**[Qwen-Image](qwen/qwen-image.md)**

Image generation · SFT and LoRA

- [Qwen-Image](qwen/qwen-image.md)

```

```{container} model-family

**[Wan](wan/index.md)**

Image-to-video and text-to-video

- [Wan2.1 I2V](wan/wan2-1-i2v.md)
- [Wan2.1 T2V LoRA](wan/wan2-1-t2v.md)

```

```{container} model-family

**[LTX](ltx/index.md)**

Audio/video LoRA and reference conditioning

- [LTX-2.3](ltx/ltx2-3.md)

```

```{container} model-family

**[MiniMax](minimax/index.md)**

First/last-frame video and audio generation

- [MiniMax H3](minimax/h3.md)

```

````

## How to use a recipe

Every model page follows the same sequence:

1. **Model introduction** — task and model-specific scope.
2. **Variants and recipes** — checkpoints and the configurations provided here.
3. **Environment and data** — dependencies, downloads, and input preparation.
4. **Launch training** — training entry point and command-line overrides.
5. **Recipe configuration** — batch size, sequence length, parallelism, and restrictions.
6. **Validation and next steps** — outputs to inspect and related features.

## Choose hardware and check a run

A provided configuration means it exists in this checkout. It does not certify
every model size, accelerator, precision, or parallelism combination. Full-size
memory requirements and convergence need their own validation. The small
[Quick Start](../get_started/quick_start.md) records its actual hardware and revision.

1. Install the [platform environment](../hardware_support/index.md) matching your checkout.
2. Read the model page, replace filesystem paths, and verify the parallel topology.
3. Start with a short run; inspect finite loss and gradient norms.
4. Check the [checkpoint completion contract](../usage/checkpoint.md#completion) before resume.
   Preprocessing-only stages produce the files described in their own recipe.

## Add a model

Follow the [model integration guide](../usage/support_new_models/guide_and_checklist.md)
and add its recipe to the appropriate family. Model names should lead to a
reader-facing guide; link the implementation and YAML from that guide.

```{toctree}
:hidden:
:maxdepth: 2

Qwen <qwen/index>
DeepSeek <deepseek/index>
Llama <llama/index>
Gemma <gemma/index>
GPT-OSS <gpt-oss/index>
Seed <seed/index>
Wan <wan/index>
LTX <ltx/index>
MiniMax <minimax/index>
```
