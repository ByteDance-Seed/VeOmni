# Eight-device training checks

Each linked YAML runs three optimizer steps with global batch size 8 on one node. Each run completed on ranks 0–7, reported finite training loss at all three steps, and wrote a distributed checkpoint with model and optimizer state. Cells show the three losses in order. An empty cell records no passing result in this check. The main support table also includes results from other environments.

The CUDA host used eight NVIDIA H20 GPUs. The Ascend host used eight 910B2 NPUs. Both ran Transformers 5.16.1. Image–text checks used ShareGPT4V conversations and confirmed an image tensor on each rank. Text checks used Tulu 3 conversations, either with the chat template or with message contents joined by newlines for plaintext training. LoRA checks load the checkpoint at its original model size and train adapters. Reduced-model entries identify narrower coverage explicitly.

| Checkpoint or model | Training scope | 8 × H20 | 8 × 910B2 |
| --- | --- | --- | --- |
| Meta-Llama-3-8B-Instruct | full tuning, text | [1.76 / 11.04 / 19.52](../../configs/models/llama3/cuda.yaml) | — |
| Meta-Llama-3.1-8B-Instruct | full tuning, text | [1.59 / 8.05 / 24.09](../../configs/models/llama3_1/cuda.yaml) | — |
| Llama-3.3-70B-Instruct | LoRA, text | — | [1.49 / 1.28 / 1.32](../../configs/models/llama3_3/ascend.yaml) |
| Qwen2-0.5B-Instruct | full tuning, text | [1.94 / 2.72 / 5.26](../../configs/models/qwen2/cuda.yaml) | [1.94 / 2.72 / 5.24](../../configs/models/qwen2/ascend.yaml) |
| Qwen2.5-0.5B-Instruct | full tuning, text | [1.95 / 2.88 / 4.44](../../configs/models/qwen2_5/cuda.yaml) | — |
| Qwen2-VL-7B-Instruct | full tuning, image + text | [12.09 / 8.78 / 8.66](../../configs/models/qwen2_vl/cuda.yaml) | [12.11 / 8.77 / 8.87](../../configs/models/qwen2_vl/ascend.yaml) |
| Qwen2.5-VL-3B-Instruct | full tuning, image + text | [12.36 / 11.49 / 11.16](../../configs/models/qwen2_5_vl/cuda.yaml) | [12.30 / 11.49 / 11.12](../../configs/models/qwen2_5_vl/ascend.yaml) |
| Qwen3-30B-A3B | LoRA, text | [1.71 / 1.51 / 1.50](../../configs/models/qwen3_moe/cuda.yaml) | [1.70 / 1.51 / 1.50](../../configs/models/qwen3_moe/ascend.yaml) |
| Qwen3-VL-30B-A3B-Instruct | LoRA, image + text | — | [1.34 / 1.33 / 1.22](../../configs/models/qwen3_vl_moe/ascend.yaml) |
| Qwen3-Omni-30B-A3B-Instruct | LoRA, image + text | [1.17 / 1.18 / 1.07](../../configs/models/qwen3_omni_moe/cuda.yaml) | [1.17 / 1.18 / 1.06](../../configs/models/qwen3_omni_moe/ascend.yaml) |
| Qwen3.5-9B-Base | full tuning, image + text | [1.31 / 1.00 / 0.81](../../configs/models/qwen3_5/cuda.yaml) | — |
| Qwen3.5-35B-A3B | LoRA, image + text | [1.32 / 1.41 / 1.18](../../configs/models/qwen3_5_moe/cuda.yaml) | — |
| Seed-OSS-36B-Base | LoRA, text | — | [1.44 / 1.08 / 1.27](../../configs/models/seed_oss/ascend.yaml) |

## Running a linked example

From the repository root, put the checkpoint at the path in the YAML and provide the indicated local data file. The first two comment lines of each YAML identify its checkpoint and launch command. Each example writes its checkpoint under `outputs/<model>/<backend>`.

For conversation or plaintext input, use a Parquet file with a `messages` column (role/content conversations) or a `text` column (strings), respectively. For image SFT, use ShareGPT4V rows with resolvable images and the configured source name. See the [Tulu data example](qwen3.md) and [image-data example](qwen3_vl.md).

The Qwen3-0.6B and Qwen3-VL-2B-Instruct eight-device results are documented in those two guides.
