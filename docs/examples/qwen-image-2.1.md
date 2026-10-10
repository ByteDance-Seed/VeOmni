# Qwen-Image-2.1 LoRA training

VeOmni supports online text-to-image LoRA training for the 7B
`Qwen/Qwen-Image-2.1` transformer. The Qwen3-VL text encoder and the
Qwen-Image-2.1 VAE stay frozen; the shipped config trains `to_q`, `to_k`,
`to_v`, and `to_out.0` adapters on one GPU.

## Install the model dependencies

Qwen-Image-2.1 support is newer than VeOmni's stable Diffusers pin. The setup
validated below keeps VeOmni's Transformers 5.16.1 pin and installs Diffusers
commit `bdc2bea37a36038c44452811610489ea30ede229` (`0.41.0.dev0`) with its
required Hugging Face Hub version:

```shell
uv pip install --upgrade \
  "huggingface-hub==1.33.0" \
  "git+https://github.com/huggingface/diffusers.git@bdc2bea37a36038c44452811610489ea30ede229"
```

The Qwen-Image-2.1 integration is registered only when the installed Diffusers
build exposes `QwenImage21Transformer2DModel`.

## Prepare the dataset

Use JSON, JSONL, CSV, Arrow, or Parquet rows containing a prompt and one target
image. For example, a JSONL row can use `text` and `image_path`:

```json
{"text":"a red seed floating above black soil","image_path":"/data/images/red-seed.png"}
```

The prompt key may be `prompt`, `text`, or `caption`. The image key may be
`image`, `image_bytes`, `image_path`, or `target_image`. Images are resized to
the configured resolution and converted to opaque RGBA because the
Qwen-Image-2.1 VAE has four input channels.

## Start a single-GPU run

The example starts at 512x512 with batch size 1, BF16 weights, gradient
checkpointing, and DDP on one GPU:

```shell
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash train.sh tasks/train_dit.py configs/dit/qwen_image21_lora.yaml \
  --model.model_path /path/to/Qwen-Image-2.1/transformer \
  --model.condition_model_path /path/to/Qwen-Image-2.1 \
  --data.train_path /path/to/train.jsonl \
  --train.checkpoint.output_dir /path/to/output
```

The adapter is exported in PEFT format under
`checkpoints/global_step_<step>/lora_ckpt/`.

## A800 80GB validation

The implementation was validated with PyTorch 2.12.1+cu130 on one NVIDIA A800
80GB. A complete `DiTTrainer` step at 512x512 produced finite loss and
gradients, exposed 33,554,432 trainable LoRA parameters, saved a DCP
checkpoint, and exported a reloadable PEFT adapter. Peak reserved memory was
30.87 GiB.

A model-contract resolution sweep kept the text encoder and VAE resident,
used batch size 1 and gradient checkpointing, and ran a full forward, backward,
and optimizer update:

| Resolution | Latent tokens | Step time | Peak reserved memory | Result |
| ---------- | ------------: | --------: | -------------------: | ------ |
| 512x512 | 1,024 | 0.52 s | 31.37 GiB | pass |
| 1024x1024 | 4,096 | 1.51 s | 33.11 GiB | pass |
| 1536x1536 | 9,216 | 3.94 s | 35.98 GiB | pass |
| 2048x2048 | 16,384 | 7.95 s | 39.99 GiB | pass |
| 3072x3072 | 36,864 | 27.21 s | 51.51 GiB | pass |
| 4096x4096 | 65,536 | 72.65 s | 67.64 GiB | pass |
| 4608x4608 | 82,944 | 111.16 s | 77.44 GiB | pass, unsafe margin |
| 4864x4864 | 92,416 | n/a | 78.27 GiB | CUDA OOM |

Use 512 or 1024 for practical single-A800 training. The 4608 result leaves
less than 2 GiB of physical headroom and is not a safe production setting.
Higher resolution, larger batches, or full-parameter training needs model
sharding or additional GPUs.

## Current scope

This path supports online text-to-image LoRA training. Image editing,
condition-image tokens, transparent-alpha-specific objectives, offline cached
latents, Ulysses/context sequence parallelism, and full-parameter training are
not implemented or validated yet.
