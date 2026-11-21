# Eight-device training checks

Each local check ran three optimizer steps with global batch size 8 on one node. Each run completed on ranks 0–7, reported finite training loss at all three steps, and wrote a distributed checkpoint with model and optimizer state. Cells show the three losses in order. An empty cell records no passing result in this check. The main support table also includes results from other environments.

Full-weight checks also inspected the loader for missing foundation-model parameters. Qwen3.5 and the eight-layer DeepSeek-V4 checkpoint use the main training tower. Their optional MTP heads are outside these recipes. Qwen-Omni uses the thinker for understanding. Its speech-generation heads are outside those checks.

The CUDA host used eight NVIDIA H20 GPUs with PyTorch 2.11. The Ascend host used eight 910B2 NPUs with torch-npu 2.10 and CANN 8.3.RC2. Both ran Transformers 5.16.1. The image-input Qwen VLM checks used ShareGPT4V conversations and confirmed an image tensor on each rank. Text checks used Tulu 3 conversations, either with the chat template or with message contents joined by newlines for plaintext training. LoRA rows train adapter parameters. The training scope identifies which checkpoints were reduced. Reduced text models use a 37-token test vocabulary in the runs initialized with the shared test tokenizer.

| Checkpoint or model | Training scope | 8 × H20 | 8 × 910B2 |
| --- | --- | --- | --- |
| BAGEL-7B-MoT | SeedOmni, image understanding | 0.82 / 0.85 / 0.76 | — |
| Cosmos3-Nano | MoT and flow connector, video + text | 0.27 / 0.30 / 0.24 | 0.27 / 0.30 / 0.25 |
| DeepSeek-V2.5 | Reduced architecture, HF backend, full tuning, text | 8.01 / 6.35 / 4.96 | 8.66 / 6.71 / 5.45 |
| DeepSeek-V3 | Reduced architecture, full tuning, text | 8.89 / 7.28 / 5.89 | 8.06 / 6.75 / 5.71 |
| DeepSeek-R1 | Reduced architecture, full tuning, text | 8.89 / 7.28 / 5.89 | 8.06 / 6.75 / 5.71 |
| DeepSeek-V4-Flash-Base-Mini-BF16 | Reduced checkpoint (eight layers), LoRA, text | 14.57 / 14.93 / 13.95 | 14.57 / 14.93 / 13.94 |
| Gemma 3 | Reduced architecture, full tuning, text | 4.88 / 4.68 / 4.43 | 4.88 / 4.71 / 4.53 |
| GLM-5 | Reduced architecture, full tuning, text | 10.30 / 8.90 / 7.73 | 10.65 / 9.15 / 8.07 |
| gpt-oss-20b-BF16 | LoRA, text | 4.14 / 4.08 / 3.76 | 4.15 / 4.08 / 3.84 |
| HunyuanVideo 1.5 | LoRA, 54-block 480p T2V checkpoint, video + text | 0.25 / 0.31 / 0.27 | 0.24 / 0.26 / 0.20 |
| Janus-1.3B | SeedOmni, image understanding | 1.02 / 1.11 / 0.96 | 1.01 / 1.11 / 0.96 |
| Meta-Llama-3-8B-Instruct | full tuning, text | 1.76 / 1.35 / 1.41 | 1.76 / 1.35 / 1.41 |
| Meta-Llama-3.1-8B-Instruct | full tuning, text | 1.59 / 1.31 / 1.36 | 1.59 / 1.31 / 1.36 |
| Llama-3.3-70B-Instruct | LoRA, text | 1.49 / 1.28 / 1.32 | 1.49 / 1.28 / 1.32 |
| LTX-2.3 | LoRA, 22B dev checkpoint, synthetic cached audio/video latents | 5.26 / 4.55 / 5.32 | 5.00 / 5.60 / 5.39 |
| MiniMax-H3 FL2VA | LoRA, synthetic cached audio/video latents | 1.83 / 11.96 / 7.33 | 2.78 / 4.73 / 10.53 |
| MiniMax-H3 Ref2VA | Reduced architecture, LoRA, synthetic cached audio/video latents | 3.68 / 7.06 / 1.47 | 4.29 / 2.48 / 1.77 |
| Qwen2-0.5B-Instruct | full tuning, text | 1.94 / 1.65 / 2.05 | 1.94 / 1.65 / 2.05 |
| Qwen2.5-0.5B-Instruct | full tuning, text | 1.95 / 1.60 / 1.94 | 1.95 / 1.60 / 1.94 |
| Qwen2-VL-7B-Instruct | full tuning, image + text | 1.18 / 0.92 / 0.77 | 1.18 / 0.92 / 0.77 |
| Qwen2.5-VL-3B-Instruct | full tuning, image + text | 1.26 / 1.21 / 0.92 | 1.26 / 1.21 / 0.93 |
| Qwen2.5-Omni-3B | full tuning, image + text | 1.61 / 1.37 / 0.97 | 1.62 / 1.37 / 0.97 |
| Qwen3-30B-A3B | LoRA, text | 1.71 / 1.51 / 1.50 | 1.70 / 1.51 / 1.50 |
| Qwen3-VL-30B-A3B-Instruct | LoRA, image + text | 1.34 / 1.33 / 1.22 | 1.34 / 1.33 / 1.22 |
| Qwen3-Omni-30B-A3B-Instruct | LoRA, image + text | 1.17 / 1.18 / 1.07 | 1.17 / 1.18 / 1.06 |
| Qwen3.5-9B-Base | full tuning, image + text | 1.31 / 1.00 / 0.81 | — |
| Qwen3.5-35B-A3B | LoRA, image + text | 1.32 / 1.41 / 1.18 | — |
| Qwen3.8-Flash-Next | Reduced architecture, full tuning, text | 4.85 / 4.85 / 4.84 | — |
| Qwen-Image | LoRA, image + text | 0.11 / 0.11 / 0.12 | 0.12 / 0.11 / 0.14 |
| Qwen-Image-Edit-2511 | LoRA, image + text | 0.11 / 0.11 / 0.12 | 0.11 / 0.10 / 0.14 |
| Seed-OSS-36B-Base | LoRA, text | 1.43 / 1.09 / 1.28 | 1.44 / 1.08 / 1.27 |
| Wan2.1-T2V-1.3B-Diffusers | full tuning, video + text | 0.64 / 0.78 / 0.60 | 1.05 / 0.87 / 0.74 |
| Wan2.2-TI2V-5B-Diffusers | full tuning, video + text | 2.27 / 2.54 / 1.81 | 2.97 / 2.57 / 2.62 |

## Coverage

This sweep covers 33 variants across 24 support-matrix rows. Qwen3 and Qwen3-VL were checked separately on both hosts. A released checkpoint may use full tuning or LoRA. The reduced-model column counts completed trainer runs at smaller sizes.

| Host | Original-size checkpoint | Reduced model, trainer | Without a passing check | Total |
| --- | ---: | ---: | ---: | ---: |
| 8 × H20 | 25 | 8 | 0 | 33 |
| 8 × 910B2 | 22 | 7 | 4 | 33 |

## Checkpoints and data used in the checks

For conversation or plaintext input, use a Parquet file with a `messages` column (role/content conversations) or a `text` column (strings), respectively. For image SFT, use ShareGPT4V rows with resolvable images and the configured source name. See the [Tulu data example](qwen3.md) and [image-data example](qwen3_vl.md).

Video DiT checks use the Tom-and-Jerry sample schema (`prompt`, `video_bytes`). Qwen-Image accepts `text` and `image_bytes`. Qwen-Image-Edit also requires a source image. See the [diffusion image guide](qwen_image_edit.md) and [video preparation](hunyuanvideo15.md). MiniMax-H3 FL2VA and Ref2VA use precomputed conditioning and latent tensors in the [offline training schema](minimax_h3.md). Ref2VA also requires reference visual tokens and their unconditional CFG counterparts. The MiniMax-H3 training row covers FL2VA and Ref2VA. The [H3-Image guide](minimax_h3_image.md) covers inference.

DeepSeek-V3 and DeepSeek-R1 share the same released architecture. Their reduced runs used the same two-layer configuration and separately executed training. The loss values describe that configuration. The DeepSeek-V2.5 small run uses `MODELING_BACKEND=hf`; the native registry lacks `deepseek_v2`.

The Qwen3.8-Flash-Next and MiniMax-H3 Ref2VA checks used locally initialized weights.

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
