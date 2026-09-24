# Qwen-Image-Edit-2511 training guide

Qwen-Image-Edit-2511 is a **text-guided image editing** model based on the
Qwen-Image DiT architecture. It takes a text instruction plus one or more source
images and generates an edited target image. The 2511 variant introduces
`zero_cond_t`: source-image latents are concatenated to the noisy target tokens
and modulated with timestep 0, so the DiT only predicts noise on the target.

## Model components

| Component | Source repo | Purpose |
|-----------|-------------|---------|
| Transformer | `Qwen/Qwen-Image-Edit-2511` | DiT backbone (has `zero_cond_t`) |
| Text encoder | `Qwen/Qwen-Image` | `Qwen2_5_VLForConditionalGeneration` (multimodal) |
| VAE | `Qwen/Qwen-Image` | `AutoencoderKLQwenImage` |
| Processor | `Qwen/Qwen-Image-Edit-2511` | `Qwen2VLProcessor` for multimodal prompt encoding |

## Prepare dataset

Each JSONL record needs a text prompt, a target image, and at least one source
image:

```json
{
    "prompt": "Make the sky red",
    "source_image": "path/to/source.jpg",
    "target_image": "path/to/target.jpg"
}
```

For multi-image editing, use a list:

```json
{
    "prompt": "Combine the two images",
    "source_images": ["path/to/src1.jpg", "path/to/src2.jpg"],
    "target_image": "path/to/target.jpg"
}
```

## Start training

```shell
bash train.sh tasks/train_dit.py configs/dit/qwen_image_edit_2511.yaml \
    --model.model_path /path/to/Qwen-Image-Edit-2511/transformer \
    --model.condition_model_path /path/to/Qwen-Image \
    --model.config_path /path/to/Qwen-Image-Edit-2511/transformer/config.json \
    --model.condition_model_cfg.processor_path /path/to/Qwen-Image-Edit-2511 \
    --data.train_path /path/to/train.jsonl \
    --data.mm_configs.data_dir /path/to/images
```

## Key config fields

| Field | Description |
|-------|-------------|
| `model.model_config.zero_cond_t` | Enable timestep-0 modulation for source tokens |
| `model.condition_model_cfg.enable_edit` | Enable edit mode (multimodal prompt + source VAE latents) |
| `model.condition_model_cfg.processor_path` | Path to `Qwen2VLProcessor` (from the edit model repo) |
| `data.source_name` | `Qwen-Image-Edit` (registers `qwen_image_edit_preprocess`) |