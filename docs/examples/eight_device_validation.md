# Eight-device training checks

Each linked YAML runs three optimizer steps with global batch size 8 on one node. Each run completed on ranks 0–7, reported finite training loss at all three steps, and wrote a distributed checkpoint with model and optimizer state. Cells show the three losses in order. An empty cell records no passing result in this check. The main support table also includes results from other environments.

Full-weight checks also inspected the loader for missing foundation-model parameters. Qwen3.5 and the eight-layer DeepSeek-V4 checkpoint use the main training tower. Their optional MTP heads are outside these recipes. Qwen-Omni uses the thinker for understanding. Its speech-generation heads are outside those checks.

The CUDA host used eight NVIDIA H20 GPUs with PyTorch 2.11. The Ascend host used eight 910B2 NPUs with torch-npu 2.10 and CANN 8.3.RC2. Both ran Transformers 5.16.1. The image-input Qwen VLM checks used ShareGPT4V conversations and confirmed an image tensor on each rank. Text checks used Tulu 3 conversations, either with the chat template or with message contents joined by newlines for plaintext training. LoRA rows train adapter parameters. The training scope identifies which checkpoints were reduced. Reduced text models use a 37-token test vocabulary where the YAML specifies the shared test tokenizer.

| Checkpoint or model | Training scope | 8 × H20 | 8 × 910B2 |
| --- | --- | --- | --- |
| BAGEL-7B-MoT | SeedOmni, image understanding | [0.82 / 0.85 / 0.76](../../configs/models/bagel/cuda.yaml) | — |
| Cosmos3-Nano | MoT and flow connector, video + text | [0.27 / 0.30 / 0.24](../../configs/models/cosmos3/cuda.yaml) | [0.27 / 0.30 / 0.25](../../configs/models/cosmos3/ascend.yaml) |
| DeepSeek-V2.5 | Reduced architecture, HF backend, full tuning, text | [8.01 / 6.35 / 4.96](../../configs/models/deepseek_v2_5/cuda.yaml) | [8.66 / 6.71 / 5.45](../../configs/models/deepseek_v2_5/ascend.yaml) |
| DeepSeek-V3 | Reduced architecture, full tuning, text | [8.89 / 7.28 / 5.89](../../configs/models/deepseek_v3/cuda.yaml) | [8.06 / 6.75 / 5.71](../../configs/models/deepseek_v3/ascend.yaml) |
| DeepSeek-R1 | Reduced architecture, full tuning, text | [8.89 / 7.28 / 5.89](../../configs/models/deepseek_r1/cuda.yaml) | [8.06 / 6.75 / 5.71](../../configs/models/deepseek_r1/ascend.yaml) |
| DeepSeek-V4-Flash-Base-Mini-BF16 | Reduced checkpoint (eight layers), LoRA, text | [14.57 / 14.93 / 13.95](../../configs/models/deepseek_v4/cuda.yaml) | [14.57 / 14.93 / 13.94](../../configs/models/deepseek_v4/ascend.yaml) |
| Gemma 3 | Reduced architecture, full tuning, text | [4.88 / 4.68 / 4.43](../../configs/models/gemma3/cuda.yaml) | [4.88 / 4.71 / 4.53](../../configs/models/gemma3/ascend.yaml) |
| GLM-5 | Reduced architecture, full tuning, text | [10.30 / 8.90 / 7.73](../../configs/models/glm5/cuda.yaml) | [10.65 / 9.15 / 8.07](../../configs/models/glm5/ascend.yaml) |
| gpt-oss-20b-BF16 | LoRA, text | [4.14 / 4.08 / 3.76](../../configs/models/gpt_oss/cuda.yaml) | [4.15 / 4.08 / 3.84](../../configs/models/gpt_oss/ascend.yaml) |
| HunyuanVideo 1.5 | LoRA, 54-block 480p T2V checkpoint, video + text | [0.25 / 0.31 / 0.27](../../configs/models/hunyuanvideo15/cuda.yaml) | [0.24 / 0.26 / 0.20](../../configs/models/hunyuanvideo15/ascend.yaml) |
| Janus-1.3B | SeedOmni, image understanding | [1.02 / 1.11 / 0.96](../../configs/models/janus/cuda.yaml) | [1.01 / 1.11 / 0.96](../../configs/models/janus/ascend.yaml) |
| Meta-Llama-3-8B-Instruct | full tuning, text | [1.76 / 1.35 / 1.41](../../configs/models/llama3/cuda.yaml) | [1.76 / 1.35 / 1.41](../../configs/models/llama3/ascend.yaml) |
| Meta-Llama-3.1-8B-Instruct | full tuning, text | [1.59 / 1.31 / 1.36](../../configs/models/llama3_1/cuda.yaml) | [1.59 / 1.31 / 1.36](../../configs/models/llama3_1/ascend.yaml) |
| Llama-3.3-70B-Instruct | LoRA, text | [1.49 / 1.28 / 1.32](../../configs/models/llama3_3/cuda.yaml) | [1.49 / 1.28 / 1.32](../../configs/models/llama3_3/ascend.yaml) |
| LTX-2.3 | LoRA, 22B dev checkpoint, synthetic cached audio/video latents | [5.26 / 4.55 / 5.32](../../configs/models/ltx2_3/cuda.yaml) | [5.00 / 5.60 / 5.39](../../configs/models/ltx2_3/ascend.yaml) |
| MiniMax-H3 FL2VA | LoRA, synthetic cached audio/video latents | [1.83 / 11.96 / 7.33](../../configs/models/minimax_h3_fl2va/cuda.yaml) | [2.78 / 4.73 / 10.53](../../configs/models/minimax_h3_fl2va/ascend.yaml) |
| MiniMax-H3 Ref2VA | Reduced architecture, LoRA, synthetic cached audio/video latents | [3.68 / 7.06 / 1.47](../../configs/models/minimax_h3_ref2va/cuda.yaml) | [4.29 / 2.48 / 1.77](../../configs/models/minimax_h3_ref2va/ascend.yaml) |
| Qwen2-0.5B-Instruct | full tuning, text | [1.94 / 1.65 / 2.05](../../configs/models/qwen2/cuda.yaml) | [1.94 / 1.65 / 2.05](../../configs/models/qwen2/ascend.yaml) |
| Qwen2.5-0.5B-Instruct | full tuning, text | [1.95 / 1.60 / 1.94](../../configs/models/qwen2_5/cuda.yaml) | [1.95 / 1.60 / 1.94](../../configs/models/qwen2_5/ascend.yaml) |
| Qwen2-VL-7B-Instruct | full tuning, image + text | [1.18 / 0.92 / 0.77](../../configs/models/qwen2_vl/cuda.yaml) | [1.18 / 0.92 / 0.77](../../configs/models/qwen2_vl/ascend.yaml) |
| Qwen2.5-VL-3B-Instruct | full tuning, image + text | [1.26 / 1.21 / 0.92](../../configs/models/qwen2_5_vl/cuda.yaml) | [1.26 / 1.21 / 0.93](../../configs/models/qwen2_5_vl/ascend.yaml) |
| Qwen2.5-Omni-3B | full tuning, image + text | [1.61 / 1.37 / 0.97](../../configs/models/qwen2_5_omni/cuda.yaml) | [1.62 / 1.37 / 0.97](../../configs/models/qwen2_5_omni/ascend.yaml) |
| Qwen3-30B-A3B | LoRA, text | [1.71 / 1.51 / 1.50](../../configs/models/qwen3_moe/cuda.yaml) | [1.70 / 1.51 / 1.50](../../configs/models/qwen3_moe/ascend.yaml) |
| Qwen3-VL-30B-A3B-Instruct | LoRA, image + text | [1.34 / 1.33 / 1.22](../../configs/models/qwen3_vl_moe/cuda.yaml) | [1.34 / 1.33 / 1.22](../../configs/models/qwen3_vl_moe/ascend.yaml) |
| Qwen3-Omni-30B-A3B-Instruct | LoRA, image + text | [1.17 / 1.18 / 1.07](../../configs/models/qwen3_omni_moe/cuda.yaml) | [1.17 / 1.18 / 1.06](../../configs/models/qwen3_omni_moe/ascend.yaml) |
| Qwen3.5-9B-Base | full tuning, image + text | [1.31 / 1.00 / 0.81](../../configs/models/qwen3_5/cuda.yaml) | — |
| Qwen3.5-35B-A3B | LoRA, image + text | [1.32 / 1.41 / 1.18](../../configs/models/qwen3_5_moe/cuda.yaml) | — |
| Qwen3.8-Flash-Next | Reduced architecture, full tuning, text | [4.85 / 4.85 / 4.84](../../configs/models/qwen3_8_flash_next/cuda.yaml) | — |
| Qwen-Image | LoRA, image + text | [0.11 / 0.11 / 0.12](../../configs/models/qwen_image/cuda.yaml) | [0.12 / 0.11 / 0.14](../../configs/models/qwen_image/ascend.yaml) |
| Qwen-Image-Edit-2511 | LoRA, image + text | [0.11 / 0.11 / 0.12](../../configs/models/qwen_image_edit/cuda.yaml) | [0.11 / 0.10 / 0.14](../../configs/models/qwen_image_edit/ascend.yaml) |
| Seed-OSS-36B-Base | LoRA, text | [1.43 / 1.09 / 1.28](../../configs/models/seed_oss/cuda.yaml) | [1.44 / 1.08 / 1.27](../../configs/models/seed_oss/ascend.yaml) |
| Wan2.1-T2V-1.3B-Diffusers | full tuning, video + text | [0.64 / 0.78 / 0.60](../../configs/models/wan2_1/cuda.yaml) | [1.05 / 0.87 / 0.74](../../configs/models/wan2_1/ascend.yaml) |
| Wan2.2-TI2V-5B-Diffusers | full tuning, video + text | [2.27 / 2.54 / 1.81](../../configs/models/wan2_2/cuda.yaml) | [2.97 / 2.57 / 2.62](../../configs/models/wan2_2/ascend.yaml) |

## Coverage

This sweep covers 33 variants across 24 support-matrix rows. Qwen3 and Qwen3-VL were checked separately on both hosts. A released checkpoint may use full tuning or LoRA. The reduced-model column counts completed trainer runs at smaller sizes.

| Host | Original-size checkpoint | Reduced model, trainer | Without a passing check | Total |
| --- | ---: | ---: | ---: | ---: |
| 8 × H20 | 25 | 8 | 0 | 33 |
| 8 × 910B2 | 22 | 7 | 4 | 33 |

## Running a linked example

Run from the repository root and provide the indicated local data file. For checkpoint-based training, put the weights at the path in the YAML. The first two comment lines of each YAML identify its checkpoint and launch command. Each example writes its checkpoint under `outputs/<model>/<backend>`.

For conversation or plaintext input, use a Parquet file with a `messages` column (role/content conversations) or a `text` column (strings), respectively. For image SFT, use ShareGPT4V rows with resolvable images and the configured source name. See the [Tulu data example](qwen3.md) and [image-data example](qwen3_vl.md).

Video DiT checks use the Tom-and-Jerry sample schema (`prompt`, `video_bytes`). Qwen-Image accepts `text` and `image_bytes`. Qwen-Image-Edit also requires a source image. See the [diffusion image guide](qwen_image_edit.md) and [video preparation](hunyuanvideo15.md). MiniMax-H3 FL2VA and Ref2VA use precomputed conditioning and latent tensors in the [offline training schema](minimax_h3.md). Ref2VA also requires reference visual tokens and their unconditional CFG counterparts. The MiniMax-H3 training row covers FL2VA and Ref2VA. The [H3-Image guide](minimax_h3_image.md) covers inference.

DeepSeek-V3 and DeepSeek-R1 share the same released architecture. Their reduced runs used the same two-layer configuration and separately executed training. The loss values describe that configuration. The DeepSeek-V2.5 small run uses `MODELING_BACKEND=hf`; the native registry lacks `deepseek_v2`.

For the two reduced examples that load generated weights, first run the corresponding command from the repository root. Both use the same deterministic initialization as the recorded run.

```bash
python scripts/prepare_reduced_model_example.py qwen3_8_flash_next
python scripts/prepare_reduced_model_example.py minimax_h3_ref2va
```

The GPT-OSS runs used the [BF16 distribution of gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-BF16/tree/5a6fe4c0c1a86147cb9dc1bb99486e61d12a3c9e). All nine local shard hashes matched that revision.

Gemma 3 was tested with the repository’s reduced text-only configuration. The published checkpoint required gated access.

The original-size HunyuanVideo check used the [480p T2V checkpoint](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038) with the online condition encoders. It trained on nine-frame clips at 256 × 384. Follow the [HunyuanVideo asset layout](hunyuanvideo15.md).

For LTX, stage the [22B dev weights](https://huggingface.co/Lightricks/LTX-2.3/tree/c1521179de480464f366a98a9d9b2acc4cc097b9) in a directory with the following layout. This is the raw upstream audio-video checkpoint. The offline run used cached features and latents. Its scope includes the embedding connectors and the 48-block transformer.

```bash
mkdir -p LTX-2.3
ln -s /absolute/path/to/ltx-2.3-22b-dev.safetensors LTX-2.3/model.safetensors
cp configs/model_configs/ltx2_3/ltx2_3_av.json LTX-2.3/config.json
```

Use the [LTX preprocessing guide](ltx-2.3.md) to produce Parquet data. The checks used synthetic cached rows with the same per-sample layout as the audio-video preprocessor, including video latents of shape `[128, F, H, W]`, audio latents of shape `[8, T, 16]`, video/audio prompt features and their mask. Raw video and audio encoding were outside the LTX run.

Cosmos uses a JSONL file with a video path and caption per row. The full configuration and supported media layout are described in the [Cosmos3 guide](../seed_omni/models/cosmos3.md). For Janus, convert the upstream DeepSeek weights to the split layout using the [Janus converter instructions](../seed_omni/models/janus.md).

The [Qwen3-0.6B](qwen3.md) and [Qwen3-VL-2B-Instruct](qwen3_vl.md) guides give the original eight-device results.

## Combinations without a passing run

| Model | Eight-device host | Observation |
| --- | --- | --- |
| BAGEL-7B-MoT | 910B2 | BAGEL requested FlexAttention, which is registered only for CUDA |
| Qwen3.5-9B-Base | 910B2 | CANN 8.3.RC2 failed to compile the GatedDeltaNet KKT kernel with Triton-Ascend 3.2.0rc4 |
| Qwen3.5-35B-A3B | 910B2 | CANN 8.3.RC2 failed to compile the GatedDeltaNet KKT kernel with Triton-Ascend 3.2.0rc4 |
| Qwen3.8-Flash-Next | 910B2 | CANN 8.3.RC2 failed to compile the GatedDeltaNet KKT kernel with Triton-Ascend 3.2.0rc4 |

A separate eight-rank Triton vector-add passed on this host. The GatedDeltaNet failure occurs in the CANN compiler when building `chunk_scaled_dot_kkt_fwd_kernel`. The alternative `fla_npu` implementation requires CANN 8.5.2 or newer on Atlas A2. See the [upstream requirements](https://github.com/flashserve/flash-linear-attention-npu/blob/a7c608f/README.md).

## Additional audio-input checks

The listed Qwen-Omni thinkers also trained on VoiceAssistant utterances in separate eight-device runs with audio features on every rank. The cells give the three training losses.

| Checkpoint | 8 × H20 | 8 × 910B2 |
| --- | --- | --- |
| Qwen2.5-Omni-3B | 2.59 / 2.32 / 2.01 | 2.59 / 2.32 / 2.01 |
| Qwen3-Omni-30B-A3B-Instruct | 2.17 / 2.28 / 2.35 | 2.16 / 2.27 / 2.35 |
