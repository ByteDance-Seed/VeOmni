# MiniMax H3 FL2VA Quick Start

This guide walks through **training** and **inference** for MiniMax H3 FL2VA (first/last frame + text -> video + audio) on an Ascend NPU machine. Every command can be copied and run directly.

- Verified environment: 4 Ascend NPUs, torch_npu + torchrun
- Verified flow: two-stage offline training (embedding -> offline) for 30 steps + single-card inference

---

## Table of Contents

1. [Model](#1-model)
2. [Data Format](#2-data-format)
3. [Training (Two Stages, Step by Step)](#3-training-two-stages-step-by-step)
4. [Training Config Notes](#4-training-config-notes)
5. [Inference](#5-inference-step-by-step)
6. [Inference Config Notes](#6-inference-config-notes)

---

## 1. Model

```shell
modelscope download --model MiniMax/MiniMax-H3 \
    --local_dir pretrained_models/MiniMax-H3
```

---

## 2. Data Format

### 2.1 Directory Layout

During Stage 1 (offline embedding), if no keyframe images are provided, the first and last frames are extracted from the video itself.

```
dataset/my_data/
├── metadata.csv          # index file (must be named metadata.csv, or set via train_path in the config)
├── video.mp4             # training video
├── first.png             # (optional) first-frame keyframe for inference
└── last.png              # (optional) last-frame keyframe for inference
```

### 2.2 metadata.csv Format

The following 4 columns are **required**, with fixed column names:

```csv
video,prompt,input_audio,frame_rate
video.mp4,"A girl is very happy, she is speaking in english.",video.mp4,24
```

| Column | Required | Meaning |
|:---|:-----|:-----|
| `video` | Yes | Video file name (relative to the CSV directory) |
| `prompt` | Yes | Text description (matches the video content) |
| `input_audio` | Yes | Audio source; set to `video.mp4` to use the video's own audio track |
| `frame_rate` | Yes | Frame rate; set to `24` |

### 2.3 Hard Video Constraints (violations raise errors)

| Constraint | Value | Notes |
|:-----|:---|:-----|
| Frame count | **124** (must satisfy `(num_frames-5) % 17 == 0`) | The Video VAE groups frames by 17; 124 is the demo-config value. 73, 107, 141, etc. are also legal (`(N-5) % 17 == 0`) |
| Resolution | **480x832** (height x width) | Must be divisible by the VAE downsampling factor |
| Frame rate | 24 | `fps: 24` in the config; audio latent length is computed as `num_frames/24*40` |
| Audio | 32kHz stereo | Audio is resampled to 32kHz automatically; videos without audio fail during training |

---

## 3. Training (Two Stages, Step by Step)

### Step 1: Run Stage 1 (offline embedding)

```shell
# MiniMax H3 FL2VA
# Offline embedding
bash train.sh tasks/train_dit.py configs/dit/minimax_h3_fl2va_embedding.yaml
```

What Stage 1 does:

- Loads Video VAE / Audio VAE / Text Encoder (**does not load the DiT**)
- Encodes video/audio/text -> VAE latents + prompt embeddings + packed-sequence info
- Writes one parquet per card: `output/minimax_h3_fl2va_embedding/rank_<rank>_shard_0.parquet`
- Success marker: process exits with no traceback

### Step 2: Switch to Stage 2 (offline training)

```shell
bash train.sh tasks/train_dit.py configs/dit/minimax_h3_fl2va_offline.yaml
```

What Stage 2 does:

- Loads the DiT (FSDP2 + gradient checkpointing + bf16), **skips** the VAE/Text Encoder (`skip_encoder_load: true`)
- Reads parquet -> adds noise -> DiT forward/backward -> AdamW update

## 4. Training Config Notes

Two config files:

| Stage | Config | Training task |
|:-----|:-----|:---------|
| Stage 1 | `configs/dit/minimax_h3_fl2va_embedding.yaml` | `training_task: offline_embedding` |
| Stage 2 | `configs/dit/minimax_h3_fl2va_offline.yaml` | `training_task: offline_training` |

### Stage 1 Config (embedding)

```yaml
model:
  condition_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
  condition_model_cfg:
    base_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
    video_vae_subfolder: video_vae/source   # VAE weights subfolder
    skip_encoder_load: false                # Stage 1 must be false (encoders are loaded)
    use_keyframe_condition: true
    keyframe_indices: [0, -1]               # first frame + last frame
    video_max_frames: 73                    # max video latent frame groups
    video_max_resolution: 848
    sigma_shift_video: 12.0
    sigma_shift_audio: 3.0

data:
  train_path: dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/metadata.csv
  data_transform: minimax_h3_online         # Stage 1 encodes raw video online
  datasets_type: minimax_h3_online
  dataloader:
    num_workers: 0                          # Stage 1 is encode-heavy; use 0 workers to avoid memory contention
    drop_last: false
  mm_configs:
    data_dir: dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/
    fps: 24
    min_frames: 124
    max_frames: 124
    height: 480
    width: 832
  offline_embedding_save_dir: output/minimax_h3_fl2va_embedding   # Stage 2 reads from here
```

**Important**:

- `train_path` must point to the **metadata.csv file**, not a directory, otherwise you get `png files are not supported`
- When changing data, `fps/min_frames/max_frames/height/width` must match the actual video parameters; the frame count must satisfy `(N-5) % 17 == 0`
- `offline_embedding_save_dir` must match Stage 2's `data.train_path`

### Stage 2 Config (offline training)

```yaml
model:
  model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer
  condition_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
  condition_model_cfg:
    skip_encoder_load: true                 # must be true: do not load VAE/TextEncoder
    video_max_frames: 120
    video_max_resolution: 832
  optimizer:
    type: adamw
    lr: 1.0e-5
    max_grad_norm: 1.0e9
  accelerator:
    init_device: meta
    gradient_checkpointing:
      enable: true                            # turning this off OOMs when memory is tight
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32

data:
  train_path: output/minimax_h3_fl2va_embedding    # output dir of Stage 1
  data_transform: dit_offline
  datasets_type: minimax_h3_offline
  shuffle: false
  mm_configs:
    repeat: 100                             # dataset repeat count (must be > 1 for small datasets)

train:
  training_task: offline_training
  global_batch_size: 8
  micro_batch_size: 1
  max_steps: 30
  checkpoint:
    output_dir: output/minimax_h3_fl2va_offline
    save_steps: 10
    save_hf_weights: false
```

---

## 5. Inference

```shell
python tasks/infer/infer_minimax_h3.py
```

The script runs two tasks sequentially:

1. **t2va**: text-only -> video + audio (480x832, 124 frames, 50 steps)
2. **fl2va**: first frame + last frame + text -> video + audio (832x480 portrait, 124 frames, 50 steps)

Output files (repo root):

- `t2va.mp4`
- `fl2va.mp4`

---

## 6. Inference Config Notes

All inference config lives in `tasks/infer/infer_minimax_h3.py`:

```python
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    condition_model_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
    condition_model_cfg={
        "base_model_path": "pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
        "use_keyframe_condition": True,
        "keyframe_indices": [0, -1],       # first and last frames
    },
    transformer_config_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
    transformer_weights_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer",
    ops_implementation=OpsImplementationConfig(
        attn_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    ),
)
```

Call parameters:

```python
# t2va
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,   # frame count must satisfy (N-5) % 17 == 0
    num_inference_steps=50, seed=0,          # fewer steps = faster; fixed seed = reproducible
)

# fl2va
video, audio = pipe(
    prompt=prompt,
    height=832, width=480, num_frames=124,
    num_inference_steps=50, seed=0,
    keyframes=[first_frame, last_frame],     # images must exist, otherwise FileNotFoundError
    keyframe_indices=[0, -1],
)
```

**Important**:

- `num_frames` must satisfy `(N-5) % 17 == 0`, otherwise the Video VAE raises an error

---
