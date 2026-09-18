# Models and Recipes

This is the entry point for model-specific training configurations. Start with
[Quick Start](../get_started/quick_start.md) if you have not run VeOmni before.

## How to read this catalog

**Provided configuration** means the YAML exists in this checkout. It does not
certify every model size, accelerator, precision, or parallelism combination.
The linked guide documents preparation and restrictions; the
[hardware guide](../hardware_support/index.md) records platform-specific evidence.
Tests using toy models check implementation behavior, not full-size memory needs
or convergence. See [testing](../testing.md) for the validation scope.

The catalog covers checked-in training recipes, not every model that the
Hugging Face fallback loader can instantiate. Choose a config for the actual
architecture and task rather than treating one family name as a compatibility promise.

## Provided configurations

| Modality / family | Workflows and configuration | Guide / restrictions |
| --- | --- | --- |
| Text: Qwen2.5 | [SFT](../../configs/text/qwen2_5.yaml) | Configure model and dataset paths; see [basic modules](../usage/basic_modules.md) |
| Text: Qwen3 dense | [SFT](../../configs/text/qwen3.yaml), [LoRA](../../configs/text/qwen3_lora.yaml), [DPO](../../configs/text/qwen3_dpo.yaml) | [SFT](qwen3.md), [DPO](qwen3_dpo.md), [LoRA](../key_features/lora.md) |
| Text: Qwen3 MoE | [SFT / EP](../../configs/text/qwen3-moe.yaml), [LoRA](../../configs/text/qwen3_moe_lora.yaml), [Muon](../../configs/text/qwen3_moe_muon.yaml) | [MoE recipe](qwen3_moe.md); choose EP size for your topology |
| Text / VLM: Qwen3.5 | [Text SFT](../../configs/text/qwen3_5_sft.yaml), [dense VLM](../../configs/multimodal/qwen3_5/qwen3_5_vl.yaml), [MoE VLM](../../configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml) | [Qwen3.5](qwen3_5.md); GPU and NPU kernel choices differ |
| Text: Qwen3.8 Flash Next / Qwen4-Exp | [SFT](../../configs/text/qwen4exp.yaml) | [PLE / EP constraints](../design/qwen4_exp_ple_2d_parallelism.md) |
| Text: Llama3 | [SFT](../../configs/text/llama3.yaml) | Configure paths and check checkpoint access before launch |
| Text: Gemma3 | [SFT](../../configs/text/gemma3.yaml) | Text recipe; this row does not claim a Gemma vision-training recipe |
| Text: DeepSeek V2/V3/R1 | [SFT](../../configs/text/deepseek.yaml), [V3 LoRA](../../configs/text/deepseek_v3_lora.yaml) | [MoE weight loading](../transformers_v5/transformers_v5_moe_weight_loading.md) |
| Text: DeepSeek V4 | [GPU SFT](../../configs/text/deepseek_v4.yaml), [NPU SFT](../../configs/text/deepseek_v4_npu.yaml), [indexer loss](../../configs/text/deepseek_v4_indexer_loss.yaml) | [Kernel selection](../design/kernel_selection.md#deepseek-v4-dsa-and-mhc), [indexer loss](../design/deepseek_v4_indexer_loss.md) |
| Text: GPT-OSS | [120B LoRA / EP4](../../configs/text/gpt_oss_120b_lora_ep4.yaml) | [LoRA guide](../key_features/lora.md); check attention backend requirements |
| Text: Seed-OSS | [Plain-text training](../../configs/text/seed_oss.yaml) | [Seed-OSS](seed_oss.md) |
| VLM: Qwen2 / Qwen2.5 VL | [Qwen2](../../configs/multimodal/qwen2_vl/qwen2_vl.yaml), [Qwen2.5](../../configs/multimodal/qwen25_vl/qwen25_vl.yaml) | [Multimodal data processing](../usage/multimodal_data_processing.md) |
| VLM: Qwen3 VL dense / MoE | [Dense](../../configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml), [MoE](../../configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml), [MoE LoRA](../../configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml) | [Qwen3 VL](qwen3_vl.md) |
| Omni: Qwen2.5 | [SFT](../../configs/multimodal/qwen25_omni/qwen25_omni.yaml) | [Multimodal data processing](../usage/multimodal_data_processing.md); use matching processor and media inputs |
| Omni: Qwen3 MoE | [SFT](../../configs/multimodal/qwen3_omni/qwen3_omni.yaml), [LoRA](../../configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml), [offline AV](../../configs/multimodal/qwen3_omni/qwen3_omni_offline_av.yaml) | [Multisource SFT](qwen3_omni_moe.md), [offline audio/video](qwen3_omni_offline_av.md) |
| Diffusion: Wan | [SFT](../../configs/dit/wan_sft.yaml), [LoRA](../../configs/dit/wan_lora.yaml), [1.3B LoRA template](../../configs/dit/wan2.1_I2V_1.3B_lora.yaml) | [I2V](wan2.1.md), [T2V LoRA with model-path overrides](wan2.1_I2V_1.3B.md) |
| Diffusion: Qwen-Image | [SFT](../../configs/dit/qwen_image_sft.yaml), [LoRA](../../configs/dit/qwen_image_lora.yaml) | [LoRA recipe](../key_features/lora.md#64-qwen-image-lora-dit-fsdp2) |
| Diffusion: LTX-2.3 | [AV LoRA](../../configs/dit/ltx2_av_lora.yaml), [low VRAM](../../configs/dit/ltx2_av_lora_low_vram.yaml), [IC-LoRA](../../configs/dit/ltx2_v2v_ic_lora.yaml) | [LTX-2.3](ltx-2.3.md); prepare the condition encoder and offline embeddings |
| Diffusion: MiniMax H3 | [Embedding](../../configs/dit/minimax_h3_fl2va_embedding.yaml), [offline training](../../configs/dit/minimax_h3_fl2va_offline.yaml) | [MiniMax H3](minimax_h3.md); guide reports a four-NPU flow |

## Choose hardware and check a run

1. Install the [platform environment](../hardware_support/index.md). Use the
   same revision for the code, documentation, and configuration.
2. Read the recipe's prerequisites and replace model, dataset, and output paths.
   A config filename alone does not specify memory requirements.
3. Start with a short run before increasing sequence length or parallelism.
   Inspect finite loss/gradient norms and the configured checkpoint outputs.
4. For checkpoint-producing runs, verify [completion metadata](../usage/checkpoint.md#completion).
   Preprocessing-only stages instead produce the files described by their recipe.

## Recipe guides

```{toctree}
:maxdepth: 1
:caption: Text

qwen3
qwen3_moe
qwen3_dpo
seed_oss
```

```{toctree}
:maxdepth: 1
:caption: Vision, language, and audio

qwen3_5
qwen3_vl
qwen3_omni_moe
qwen3_omni_offline_av
```

```{toctree}
:maxdepth: 1
:caption: Diffusion

wan2.1
wan2.1_I2V_1.3B
ltx-2.3
minimax_h3
```
