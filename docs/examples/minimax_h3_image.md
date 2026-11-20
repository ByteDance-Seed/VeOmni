# MiniMax-H3-Image inference and fine-tuning

MiniMax-H3-Image is the image-generation member of the MiniMax-H3 family.
It reuses the H3 text encoder and the H3 diffusion transformer without the
audio branch, samples on a fixed six-step schedule, and supports text-to-image
and reference-image editing.

## Download

```shell
hf download MiniMaxAI/MiniMax-H3-Image --local-dir pretrained_models/MiniMax-H3-Image \
  --include 'transformer/*' --include 'text_encoder/*' \
  --include 'vae_diffusers/*' --include 'processor/*' \
  --include 'pdd_fused_heads.safetensors' --include 'pdd_config.json'
```

The checkpoint directory contains:

```text
MiniMax-H3-Image/
├── transformer/                 # native MiniMax-H3 DiT layout (audio disabled)
├── vae_diffusers/               # image VAE in diffusers layout
├── text_encoder/                # Qwen3-VL, read at layer 50
├── processor/
├── pdd_fused_heads.safetensors  # one output head per sampling step
└── pdd_config.json
```

## Text-to-image

The pipeline requires `audio_enabled=False` (the default). An
`audio_enabled=True` transformer override is rejected before loading weights.

```python
import torch
from veomni.models.diffusers.minimax_h3_image.inference import MiniMaxH3ImagePipeline

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    "pretrained_models/MiniMax-H3-Image",
    torch_dtype=torch.bfloat16,
)
image = pipe(
    prompt="A quiet mountain lake at sunrise, photorealistic, wide angle",
    height=2048,
    width=2048,
    seed=0,
)
image.save("t2i.png")
```

## Image editing

References are referred to as `<Picture 1>`, `<Picture 2>`, ... in the prompt.
Each reference is encoded at its own size; only the generated image uses the
requested canvas.

```python
from PIL import Image

reference = Image.open("reference.png")
image = pipe(
    prompt="<Picture 1> The same character, now standing in a snowy forest at night",
    height=2048,
    width=2048,
    references=[reference],
    seed=0,
)
image.save("edit.png")
```

## Text-to-image fine-tuning

The online and offline paths support `micro_batch_size: 1` and `ulysses_size: 1`.
They follow the public [DiffSynth fine-tuning recipe](https://github.com/modelscope/DiffSynth-Studio/blob/fc8b4be56759de06cac585929f608227ac58766b/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image.sh): the DiT learns a flow-matching target while the released PDD heads remain frozen. Image training uses the same H3 layout and DiT `forward()` as video, with one latent frame and no audio tokens. The image VAE and noise schedule are still image-specific.

Both paths sample fresh noise and timesteps. The 1,000-step training schedule uses `shift=2.22`; each training sigma selects the nearest released sigma except the terminal zero. The resulting head buckets have unequal training weight, and their ranges extend beyond the exact inference sigmas. This is flow-matching fine-tuning, not a reproduction of the original PDD distillation. Reference-image editing uses the same objective; multi-sample packing and Ulysses SP are not supported by these recipes.

### Prepare weights and data

After downloading the release, prepare a training transformer directory. Preparation reads the number of heads from `pdd_fused_heads.safetensors` and requires an explicit, matching `sigmas` schedule in `pdd_config.json`. Only inference can fall back to the legacy schedule if it is missing. The original shards are linked by absolute path, not copied. Keep the release and its shards accessible at those paths on every training node.

```shell
uv run python -m scripts.prepare_minimax_h3_image_t2i \
  --model-dir pretrained_models/MiniMax-H3-Image \
  --out-dir pretrained_models/MiniMax-H3-Image-t2i/transformer
```

When migrating an older prepared checkpoint, prepare it again in a new directory. In both the online and offline training YAMLs, update `model.model_path` to that directory, `model.config_path` to its `config.json`, and `model.condition_model_path` to its `condition.json` together. The embedding YAML does not load the DiT weights; update only `model.config_path` and `model.condition_model_path` there.

The example expects a CSV with `image,prompt` columns and paths relative to `data.mm_configs.data_dir`. The following pinned DiffSynth sample contains five images:

```shell
dataset_url=https://modelscope.cn/datasets/DiffSynth-Studio/diffsynth_example_dataset/resolve/e1b8bcbb91db4097e6b595179238cb76556c8233/minimax_h3_image/MiniMax-H3-Image
mkdir -p dataset/minimax_h3_image
for file in metadata.csv 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg; do
  curl -fsSL "$dataset_url/$file" -o "dataset/minimax_h3_image/$file" || exit 1
done
```

### Offline training: cache and run one step

Cache first on a device with enough room to hold the truncated BF16 text encoder (about 48 GiB of parameters), FP32 VAE, and their activations. The embedding run does not load the DiT. The cache stores text embeddings, token tags, and normalized VAE posterior modes. It contains serialized tensors; only load a cache you trust.

For a training target, bilinear cover resizing preserves the source aspect ratio before center-cropping to the configured `height` and `width`. The crop fixes the canvas and can cut content near the edges. Training uses the VAE posterior mode so cache creation is deterministic and does not consume training noise RNG. Inference references serve a different purpose: they are resized by short-edge bounds with bicubic interpolation to preserve their composition, and use posterior samples as in the release.

The eight-rank recipe below needs at least eight cached rows to form a complete data-parallel batch. Before running the embedding command, duplicate the five *data rows* in the CSV once, keeping a single header. If you already cached five rows, editing the CSV is not enough: rerun the embedding stage with the expanded CSV into a fresh `data.offline_embedding_save_dir` and point the offline training YAML's `data.train_path` there.

```shell
NPROC_PER_NODE=1 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_t2i_embedding.yaml
```

The second stage loads the DiT without the text encoder or VAE. The reference config uses FP32 throughout to preserve the release projections. With about 33B parameters, weights, gradients, and the two AdamW moments alone account for about 62 GiB per rank across eight ranks, before activations and FSDP2 all-gathers. This is a memory estimate, not a measured peak. The meta-initialized FSDP2 config needs at least two ranks. Set the global batch size to the number of data-parallel ranks for this smoke run.

```shell
NPROC_PER_NODE=8 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_t2i_offline.yaml \
  --train.global_batch_size 8 --train.max_steps 1
```

The smoke config does not save weights. For longer runs, set a positive `train.checkpoint.save_steps` to save a DCP training checkpoint. The example does not export a new six-step inference checkpoint.

### Online training: encode during each step

Use the prepared transformer and the same `image,prompt` CSV directly. The online path calls the frozen text encoder and VAE on every sample, then uses the same condition processing and loss as offline training. No embedding-cache stage is needed.

The meter sees the raw prompts and images before encoding, so it reports zero MFU, token throughput, and `consumed_chunk_num` for this online recipe. This does not indicate a failed training step.

The CUDA sync test checks for implicit synchronization calls. Copies of CPU noise, schedules, and input data can still stall during staging from pageable memory; passing this test does not establish copy/compute overlap.

The online reference keeps the same FP32 DiT training policy as the offline config and additionally holds about 50.2 GiB of frozen encoder/VAE parameters on each rank. For eight ranks, the weights, gradients, and AdamW moments estimated above plus those encoders exceed 112 GiB per rank before activations and communication buffers. Use hardware with sufficient memory or increase the shard count; this full-parameter recipe does not fit an eight-rank 64 GiB node. Tiny-DiT online validation on a single 64 GiB Ascend NPU does not establish released-33B training capacity.

As with offline training, the CSV must contain at least one row per data-parallel rank. If it still contains only the original five sample rows, duplicate them once and keep one header before running the eight-rank command.

```shell
NPROC_PER_NODE=8 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_t2i_online.yaml \
  --train.global_batch_size 8 --train.max_steps 1
```

Both training configs use `dataset_repeat: true` and can run more updates by increasing `train.max_steps`. Set `train.checkpoint.save_steps` to save training checkpoints during a longer run. The online example uses its own output directory, `output/minimax_h3_image_t2i_online_sft`.

## Reference-image editing fine-tuning

Reuse the prepared transformer and its paired `config.json` and `condition.json` from the T2I recipe. The input CSV has `image,prompt,reference_images` columns. The `image` is the target, and `reference_images` is a JSON array of one or more paths. Paths are relative to `data.mm_configs.data_dir`.

The pinned DiffSynth editing sample below has one target and two references. For an eight-rank smoke test, write eight copies of its data row **before** generating the cache. For your own dataset, provide at least one row per data-parallel rank. If you already cached fewer rows, rerun the embedding stage into a fresh `data.offline_embedding_save_dir` and update the offline YAML's `data.train_path`.

```shell
dataset_url=https://modelscope.cn/datasets/DiffSynth-Studio/diffsynth_example_dataset/resolve/e1b8bcbb91db4097e6b595179238cb76556c8233/minimax_h3_image/MiniMax-H3-Image-Edit
data_dir=dataset/minimax_h3_image_edit
mkdir -p "$data_dir/edit"
for file in metadata.json edit/image1.jpg edit/image_color.jpg edit/image2.jpg; do
  curl -fsSL "$dataset_url/$file" -o "$data_dir/$file" || exit 1
done
uv run python - "$data_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
(sample,) = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
with (data_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["image", "prompt", "reference_images"])
    writer.writerows([[sample["image"], sample["prompt"], json.dumps(sample["edit_image"])]] * 8)
PY
```

References are resized as in inference, appear in the text encoder's visual prompt, and are encoded as normalized VAE posterior modes for training. The shared DiT applies the inference reference timestep of `0.999` to the reference rows and computes weighted MSE only over the target rows. Only load trusted caches.

```shell
NPROC_PER_NODE=1 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_i2i_embedding.yaml
NPROC_PER_NODE=8 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_i2i_offline.yaml
```

The online recipe reads the same expanded raw CSV directly without generating a cache. As in T2I, the online meter sees inputs before encoding and reports zero MFU, token throughput, and `consumed_chunk_num`; this does not indicate a failed step.

The CUDA sync check covers VeOmni's H3 condition and DiT code, not the delegated Qwen3-VL vision encoder. Pageable host-memory transfers can still stall during staging.

```shell
NPROC_PER_NODE=8 uv run bash train.sh tasks/train_dit.py configs/dit/minimax_h3_image_i2i_online.yaml
```

The [T2I memory estimates](#offline-training-cache-and-run-one-step) do not include the extra tokens and VAE work for each reference. Full-parameter released-checkpoint editing training has not been validated.

## Notes

- The released checkpoint is distilled to six steps with no classifier-free
  guidance. `num_inference_steps` is therefore not exposed.
- Both sides of the generated canvas are multiples of 32. Requests below
  `min_generation_area` (default `2048 * 2048`) are generated on a larger
  canvas and resized back.
- The text encoder and DiT are about 66 GB each. The pipeline stages the text
  encoder and VAE on demand and keeps them off the device while the DiT runs.
- For custom VAE tiling, each multi-tile encoding overlap must be at least one
  latent unit (16 image pixels). Tile sizes round down and overlaps round up to
  the 16-pixel latent grid, keeping encode/decode dimensions aligned. Smaller
  overlap settings remain available for decoding and single-tile encoding.
