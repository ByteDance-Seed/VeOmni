# LTX-2.3 training guide

## Download model

Download the LTX-2.3 transformer weights and the Gemma3 text encoder:

```shell
# LTX-2.3 transformer weights
python3 scripts/download_hf_model.py \
    --repo_id Lightricks/LTX-2.3 \
    --local_dir /path/to/ltx-2.3-model.safetensors

# Gemma3 text encoder (required for conditioning)
python3 scripts/download_hf_model.py \
    --repo_id google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local_dir /path/to/gemma3-model
```

## Prepare Dataset

Use the built-in preprocessing pipeline to split videos, generate captions, and compute latents/embeddings:

### Step 1: Split scenes (optional)

Split raw videos into scene clips using PySceneDetect:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py split-scenes \
    --video_dir /path/to/raw/videos \
    --output_dir /path/to/output/clips
```

### Step 2: Generate captions

Auto-caption video clips using a multimodal model (Qwen2.5-Omni by default):

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py caption \
    --input_dir /path/to/output/clips \
    --output /path/to/output/dataset.json
```

### Step 3: Compute text embeddings and VAE latents

Compute text embeddings + VAE latents from the dataset file:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py preprocess \
    --dataset_file /path/to/output/dataset.json \
    --gemma_model_path /path/to/gemma3-model \
    --checkpoint_path /path/to/ltx-2.3-model.safetensors \
    --resolution_buckets "960x544x49"
```

### Step 4: Generate reference videos (optional, for IC-LoRA)

Generate Canny edge reference videos for IC-LoRA training:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py compute-reference \
    --input_dir /path/to/output/clips \
    --dataset_file /path/to/output/dataset.json
```

### Step 5: Pack precomputed files

Pack precomputed `.pt` files into parquet shards for offline training:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py save-parquet \
    --precomputed_dir /path/to/output/.precomputed \
    --output_dir /path/to/parquet_output \
    --pad_to_multiple_of 8
```

Output directory structure:

```
data_dir/
├── .precomputed/
│   ├── audio_latents/        # Audio latents (optional, for AV training)
│   ├── conditions/           # Gemma text embeddings
│   ├── latents/              # VAE-encoded video latents
│   └── reference_latents/    # Reference video latents (optional, for Video-to-Video training)
├── clips/                    # Scene-split video clips (from split-scenes)
├── parquet_output/           # Parquet shards (from save-parquet)
│   ├── shard_0000.parquet
│   ├── shard_0001.parquet
│   └── ...
└── dataset.json              # Captions + video paths
```

## Update config paths

Before training, update the model and data paths in the config file:

```yaml
# configs/dit/ltx2_av_lora.yaml
model:
  model_path: "/path/to/ltx-2.3-model.safetensors"
  condition_model_path: "/path/to/ltx-2.3-checkpoint"
  condition_model_cfg:
    gemma_model_path: "/path/to/gemma3-model"

data:
  train_path: "/path/to/preprocessed/data"
```

## Start training

### Audio-Video LoRA (default)

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_av_lora.yaml
```

### Audio-Video LoRA (Low VRAM)

For GPUs with limited VRAM, use the low-memory configuration with reduced LoRA rank (16 vs 32):

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_av_lora_low_vram.yaml
```

### Video-to-Video (IC-LoRA)

For video-to-video transformations (e.g., depth-to-video, style transfer), use the IC-LoRA configuration. This requires reference videos in your dataset:

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_v2v_ic_lora.yaml
```
