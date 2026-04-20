# Multimodal Data Processing

This guide explains how VeOmni processes image and video inputs for vision-language model training, including resolution control, frame sampling, and the dynamic video pixel budget introduced for Qwen3-VL.

## Overview

The multimodal data pipeline lives in `veomni/data/multimodal/` and handles:

1. **Preprocessing** (`preprocess.py`) — converts raw data samples into a unified conversation format via the [Preprocessor Registry](../../key_features/preprocessor_registry.md).
2. **Image processing** (`image_utils.py`) — loads images, resizes them to fit pixel budgets while preserving aspect ratio and ViT patch alignment.
3. **Video processing** (`video_utils.py`) — loads videos (via torchcodec), samples frames by FPS, resizes spatially, and optionally extracts audio.
4. **Transform** (`multimodal_transform.py`) — orchestrates the above into tokenized model inputs with proper masking.

All processing parameters are configured through the `mm_configs` section in your YAML config.

## Image Resolution Control

Images are resized by `smart_resize` to satisfy three constraints simultaneously:

| Parameter | Description |
|-----------|-------------|
| `image_min_pixels` | Minimum total pixels (H x W) after resize |
| `image_max_pixels` | Maximum total pixels (H x W) after resize |
| `scale_factor` | Align H and W to multiples of this (e.g., 28 for Qwen-VL ViT with patch_size=14, merge_size=2) |
| `max_ratio` | Maximum allowed aspect ratio (max_dim / min_dim) |

The resize preserves aspect ratio: it scales H and W by the same factor, then rounds to `scale_factor` multiples.

**Example config:**
```yaml
data:
  mm_configs:
    image_max_pixels: 602112  # 28 * 28 * 768
    # image_min_pixels: 3136  # optional lower bound
```

## Video Frame Sampling

Video processing has two stages: **temporal sampling** (selecting which frames to keep) and **spatial resize** (adjusting resolution per frame).

### Temporal Sampling Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fps` | Target sampling FPS | 2.0 |
| `min_frames` | Minimum output frames | None |
| `max_frames` | Maximum output frames | None |
| `frame_factor` | Align frame count to multiples of this | None |
| `frame_factor_remainder` | Remainder for frame alignment (e.g., 1 for counts like 1, 5, 9, 13...) | 0 |

Frame sampling flow:
1. Compute target frame count: `nframes = total_frames / video_fps * fps`
2. Clamp to `[min_frames, max_frames]`
3. Align to `frame_factor` (round down)
4. Uniformly sample `nframes` indices from the video
5. Pad with last frame if `nframes > total_frames`

### Spatial Resize Parameters

| Parameter | Description |
|-----------|-------------|
| `video_min_pixels` | Minimum per-frame pixels (H x W) |
| `video_max_pixels` | Maximum per-frame pixels (H x W) |
| `scale_factor` | Align H and W to multiples of this |

**Example config:**
```yaml
data:
  mm_configs:
    video_max_pixels: 602112  # 28 * 28 * 768
    max_frames: 16
    fps: 2.0
```

## Dynamic Video Pixel Budget (`video_total_pixels`)

> Introduced for Qwen3-VL. This is a no-op for models that don't set it.

### Problem

With a fixed `video_max_pixels`, every frame gets the same maximum resolution regardless of how many frames there are. A 4-frame video and a 64-frame video would each have frames at the same resolution, causing long videos to produce far more visual tokens and potentially exceed the model's context window.

### Solution

`video_total_pixels` sets a **total pixel budget across all frames**. Before spatial resizing, the per-frame `video_max_pixels` is dynamically adjusted:

```
dynamic_max = video_total_pixels / nframes * temporal_merge_factor
dynamic_max = min(dynamic_max, video_max_pixels)       # don't exceed original cap
dynamic_max = max(dynamic_max, video_min_pixels * 1.05) # don't go below minimum
```

This mirrors the official `qwen-vl-utils` logic: more frames -> lower per-frame resolution, keeping total visual tokens predictable.

### How to Set `video_total_pixels`

The formula is:

```
video_total_pixels = max_seq_len * (patch_size * merge_size)^2 * budget_ratio
```

Where:
- `max_seq_len`: your training sequence length (e.g., 4096)
- `patch_size * merge_size`: ViT spatial granularity (14 * 2 = 28 for Qwen-VL family)
- `budget_ratio`: fraction of context reserved for visual tokens (0.9 = 90%)

**Example** (max_seq_len=4096):
```
4096 * 28^2 * 0.9 = 4096 * 784 * 0.9 ≈ 2,889,523
```

For inference-scale contexts (128K tokens):
```
128000 * 784 * 0.9 ≈ 90,316,800
```

### Config Example

```yaml
data:
  max_seq_len: 4096
  mm_configs:
    image_max_pixels: 602112
    video_max_pixels: 602112
    video_total_pixels: 2889523  # dynamic per-frame budget
    max_frames: 16
    fps: 2.0
    use_audio_in_video: false
```

### Behavior Summary

| Scenario | `video_total_pixels` absent | `video_total_pixels` present |
|----------|---------------------------|------------------------------|
| 4 frames | Each frame up to 602112 px | Each frame up to min(602112, 2889523/4*2) ≈ 602112 px |
| 16 frames | Each frame up to 602112 px | Each frame up to min(602112, 2889523/16*2) ≈ 361190 px |
| 64 frames | Each frame up to 602112 px | Each frame up to min(602112, 2889523/64*2) ≈ 90297 px |

With `video_total_pixels`, longer videos automatically get lower per-frame resolution to stay within the token budget.

## Audio Processing

When `use_audio_in_video: true`, audio is extracted from video files and resampled to the target sample rate (default 16kHz). Audio processing parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_audio_in_video` | Whether to extract audio from videos | false |
| `sample_rate` | Target audio sample rate | 16000 |

## Full `mm_configs` Reference

```yaml
data:
  mm_configs:
    # Image
    image_min_pixels: 3136        # optional, min pixels per image
    image_max_pixels: 602112      # max pixels per image
    scale_factor: 28              # ViT patch alignment (patch_size * merge_size)
    max_ratio: 200                # max aspect ratio

    # Video - temporal
    fps: 2.0                      # target sampling FPS
    min_frames: 4                 # optional, minimum frames
    max_frames: 16                # maximum frames
    frame_factor: 2               # align frame count to multiples of this
    frame_factor_remainder: 0     # remainder for frame alignment

    # Video - spatial
    video_min_pixels: 3136        # optional, min pixels per frame
    video_max_pixels: 602112      # max pixels per frame
    video_total_pixels: 2889523   # optional, total pixel budget (Qwen3-VL)

    # Audio
    use_audio_in_video: false
    sample_rate: 16000
```
