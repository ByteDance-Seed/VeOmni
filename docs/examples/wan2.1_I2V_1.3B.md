# Wan2.1-T2V Training Guide

This guide covers LoRA fine-tuning of [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) using VeOmni, including dataset preparation, multi-GPU training with Ulysses Sequence Parallelism (SP), and inference with trained adapters.

---

## 1. Environment Setup

```shell
uv sync --extra gpu --dev
source .venv/bin/activate
```

For inference, install the video I/O backend:

```shell
pip install imageio imageio-ffmpeg
```

---

## 2. Download Model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local_dir ./Wan2.1-T2V-1.3B-Diffusers
```

---

## 3. Prepare Dataset

VeOmni supports two training workflows:

| Workflow | `training_task` | Description |
|---|---|---|
| **Offline** (recommended) | `offline_training` | Pre-embed videos once; re-use embeddings across epochs. Saves GPU memory during training. |
| **Online** | `online_training` | Embed videos on-the-fly each step. Requires the VAE + text encoder to stay on GPU throughout training. |

### 3.1 Download the Tom and Jerry dataset

This guide uses [Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset](https://huggingface.co/datasets/Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset) (~6 000 video clips, 540×360, 6 s each).

```shell
python3 scripts/download_hf_model.py \
    --repo_id Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset \
    --repo_type dataset \
    --local_dir ./Tom-and-Jerry-VideoGeneration-Dataset
```

The downloaded directory has the following structure:

```
Tom-and-Jerry-VideoGeneration-Dataset/
├── captions.txt   # one caption per line
├── videos.txt     # one relative video path per line (mirrors captions.txt)
└── videos/        # video files
```

### 3.2 Convert to VeOmni Parquet format

The conversion script reads `captions.txt` and `videos.txt`, loads each video as raw bytes, and writes sharded Parquet files (`0.parquet`, `1.parquet`, …) with columns `prompt`, `video_bytes`, and `source`.

```shell
python3 scripts/multimodal/convert_data/tom-and-jerry.py \
    --dataset_path ./Tom-and-Jerry-VideoGeneration-Dataset \
    --output_dir   ./Tom-and-Jerry-VideoGeneration-Dataset-parquet
```

### 3.3 Offline Workflow (recommended)

#### Step 1 – Run offline embedding (once)

This step encodes every video with the VAE and every caption with the T5 text encoder, saving the embeddings as Parquet shards. It only needs to run once per dataset.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/wan2.1_I2V_1.3B_lora.yaml \
    --model.model_path           ./Wan2.1-T2V-1.3B-Diffusers/transformer \
    --model.condition_model_path ./Wan2.1-T2V-1.3B-Diffusers \
    --data.train_path            ./Tom-and-Jerry-VideoGeneration-Dataset-parquet \
    --data.source_name           Tom-and-Jerry-VideoGeneration-Dataset \
    --data.offline_embedding_save_dir ./Tom-and-Jerry-VideoGeneration-Dataset_offline \
    --train.training_task        offline_embedding \
    --train.global_batch_size    4 \
    --train.accelerator.ulysses_size 1
```

The resulting `Tom-and-Jerry-VideoGeneration-Dataset_offline/` directory contains `rank_N_shard_M.parquet` files. Each row stores two pickled tensors:

| Column | Shape | Description |
|---|---|---|
| `latents` | `(1, 32, F, H, W)` | VAE posterior parameters (mean + log-variance concatenated along the channel axis; `32 = 2 × 16`) |
| `context` | `(1, 512, 4096)` | T5 text embedding |

#### Step 2 – Train on the offline dataset

```shell
SP_SIZE=2
NPROC_PER_NODE=8   # 4 DP replicas × SP_SIZE=2

bash train.sh tasks/train_dit.py configs/dit/wan2.1_I2V_1.3B_lora.yaml \
    --model.model_path           ./Wan2.1-T2V-1.3B-Diffusers/transformer \
    --model.condition_model_path ./Wan2.1-T2V-1.3B-Diffusers \
    --data.train_path            ./Tom-and-Jerry-VideoGeneration-Dataset_offline \
    --data.source_name           Tom-and-Jerry-VideoGeneration-Dataset \
    --train.training_task        offline_training \
    --train.global_batch_size    8 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size ${SP_SIZE} \
    --train.checkpoint.output_dir ./exp/Wan2.1-T2V-1.3B-Diffusers_lora \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.save_epochs 5 \
    --train.checkpoint.load_path auto \
    --train.num_train_epochs 30 \
    --train.wandb.enable false
```

### 3.4 Online Workflow

Pass raw Parquet videos directly during training. The VAE and text encoder run each step.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/wan2.1_I2V_1.3B_lora.yaml \
    --model.model_path           ./Wan2.1-T2V-1.3B-Diffusers/transformer \
    --model.condition_model_path ./Wan2.1-T2V-1.3B-Diffusers \
    --data.train_path            ./Tom-and-Jerry-VideoGeneration-Dataset-parquet \
    --data.source_name           Tom-and-Jerry-VideoGeneration-Dataset \
    --data.mm_configs.fps        24 \
    --data.mm_configs.max_frames 81 \
    --train.training_task        online_training \
    --train.global_batch_size    4 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size 1 \
    --train.checkpoint.output_dir ./exp/Wan2.1-T2V-1.3B-Diffusers_lora \
    --train.checkpoint.save_hf_weights true \
    --train.num_train_epochs 30
```

---

## 4. Training Configuration

The default LoRA config (`configs/dit_new/wan_lora.yaml`) targets the attention and feed-forward projections:

```yaml
model:
  lora_config:
    rank: 128
    alpha: 64
    lora_modules:
      - to_q
      - to_k
      - to_v
      - to_out.0
      - ffn.net.0.proj
      - ffn.net.2
```

### Sequence Parallelism (SP)

VeOmni supports Ulysses SP for long video sequences. SP splits the sequence dimension across GPUs within each data-parallel replica, reducing per-GPU memory while keeping training numerically equivalent to SP=1.

| `ulysses_size` | GPUs (with 4 DP replicas) |
|---|---|
| 1 | 4 |
| 2 | 8 |

Set `--train.accelerator.ulysses_size` to enable SP. The loss and gradient norms are aligned between SP=1 and SP=2 at equal DP sizes.

---

## 5. Checkpoint Output

When `--train.checkpoint.save_hf_weights true` is set, each save produces a directory compatible with `load_lora_adapter`:

```
exp/Wan2.1-T2V-1.3B-Diffusers_lora/checkpoints/
└── global_step_200/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## 6. Inference

### 6.1 Base model (no LoRA)

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "./Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = (
    "Tom, the mischievous gray cat, is sprawled out on a vibrant red pillow, "
    "his body relaxed and his eyes half-closed, as if he's just woken up or is "
    "about to doze off. His white paws are stretched out in front of him, and his "
    "tail is casually draped over the edge of the pillow."
)
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output.mp4", fps=15)
```

### 6.2 With trained LoRA adapter

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "./Wan2.1-T2V-1.3B-Diffusers"
lora_dir = "./exp/Wan2.1-T2V-1.3B-Diffusers_lora/checkpoints/global_step_200"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

pipe.transformer.load_lora_adapter(lora_dir, prefix="base_model.model", adapter_name="wan_lora")
pipe.set_adapters("wan_lora", adapter_weights=1.0)  # adjust strength between 0.5–1.0

prompt = "..."
negative_prompt = "..."

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]

export_to_video(output, "output_lora.mp4", fps=15)
```
