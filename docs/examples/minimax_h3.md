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
5. [Inference](#5-inference)
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

```text
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
  datasets_type: mapping
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

- `train_path` must point to the **metadata.csv file** (or a directory that contains it). Directories are scanned for `parquet` / `json` / `csv` / `arrow` files only, so leftover images or `veomni_cli.yaml` are ignored.
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
  datasets_type: iterable
  shuffle: false
  dataset_repeat: true

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

## Packed Offline Training

With `train.micro_batch_size > 1`, H3 packs samples inside its ordinary forward.
Both **FL2VA** and **visual Ref2VA** (image/video references, without reference
audio) use `process_condition(**batch) → model(**batch)` and return sample-mean
scalar losses. There is no Trainer packing switch or alternate output protocol.
The recipe defaults to one sample. This does not batch inference requests,
enable dynamic batching, load new encoders, or implement an RL objective.

Use the offline recipe with the updated packed metadata:

```shell
bash train.sh tasks/train_dit.py configs/dit/minimax_h3_fl2va_offline.yaml \
  --train.micro_batch_size 2 \
  --train.global_batch_size 16
```

Keep `train.dyn_bsz=false`, `data.dataloader.drop_last=true`, and FSDP2
`mixed_precision.cast_forward_inputs=false`. Timesteps stay FP32 and positions
stay FP32/FP64; blanket BF16 input casting is rejected rather than silently
changing their precision. Samples in one microbatch may differ in task
(FL2VA/visual Ref2VA, with or without keyframes), target video/audio geometry,
prompt length, reference count and reference geometry; `DiTDataCollator` fills
keys a sample lacks with `None`, and multi-sample outputs return per-sample
prediction lists. Multi-sample packing rejects Ulysses SP and block/checkpoint
offload inside modeling. Single-device/FSDP2 with SP/CP/TP/PP sizes one is the validation target;
LoRA, compilation and additional parallel/offload combinations are not validated.

FL2VA and Ref2VA layouts now contain exactly `[text | cond | audio | video]`, with
`seq_len=used` and `cu_seqlens=[0, used]`. There is no 64-row tail to crop during
batch packing. Regenerate old cached `packed` metadata with the current builders
before multi-sample training; latent tensors and embeddings need not be re-encoded.
Legacy padded metadata is rejected for multi-sample packing. Any divisibility
padding for single-sample Ulysses remains local to `MiniMaxH3DiT.forward`.
Uncovered SP attention rows are zero-initialized so discarded outputs cannot
introduce nonfinite parameter gradients.

Samples encoded without an audio track keep the silent placeholder latent, so the
layout is unchanged, but carry `has_audio=False` (also saved in offline
embeddings). Their `mse_audio` is zero-weighted, and a packed microbatch takes
the plain sample mean, so after the trainer's division by the accumulation steps
every sample keeps weight `1/G`, as with `micro_batch_size=1`. Normalizing by the
global audio-sample count is not implemented. Caches written without `has_audio`
keep supervising audio as before.

### Visual Ref2VA prepared data

Use matching Ref2VA model weights and **precomputed** Ref2VA prompt/condition
embeddings. The existing FL2VA embedding recipe does not become a Ref2VA encoder.
A decoded offline sample has the usual `input_latents`, `audio_input_latents`,
`prompt_embeds` and `use_gradient_checkpointing`, plus:

```python
from veomni.models.diffusers.minimax_h3.minimax_h3_core.packed_sequence import build_packed_ref2va

# Illustrative geometry; match it to the actual encoded tensors.
ref_blocks = [
    {"kind": "image", "latent_t": 1, "latent_h": 16, "latent_w": 24},
    {"kind": "video", "latent_t": 6, "latent_h": 16, "latent_w": 24},
]
packed = build_packed_ref2va(
    text_len=prompt_embeds.shape[0],
    latent_t=video_latents.shape[2],
    latent_h=video_latents.shape[3],
    latent_w=video_latents.shape[4],
    audio_t=audio_latents.shape[-1],
    audio_channel=audio_latents.shape[0],
    ref_blocks=ref_blocks,
    text_token_tags=text_token_tags,
)
```

Store this `packed` dictionary and `ref_visual_anchor` of shape
`[packed["cond_rows"], 96]` in the existing offline-record format. Anchor rows must
be concatenated in reference-block order and already use the same conditioning
noise augmentation as `model.condition_model_cfg.imgvid_cond_noise_aug`, matching
the native inference reference encoder. Preserve the Ref2VA presentation's text
versus vision token tags. Do not substitute target-video keyframes or FL2VA prompt
embeddings for Ref2VA references. Reference audio is rejected explicitly.

Each sample samples its own timestep/noise through the existing condition path
before packing. The transformer then executes once over compact rows with
independent main-DiT and text-refiner cumulative boundaries. Its RoPE coordinates
stay sample-local; timestep tables are remapped, not assumed shared. Reference
rows are cropped separately for each output. Video/audio signs, unpatchification,
scheduler weights and sample-mean losses retain their single-sample meanings.
For multi-sample inputs, the small text token refiner runs separately for each sample. Its BF16 output
projections can round differently when their GEMM row count changes; the deep
pretrained DiT amplifies those differences. Keeping the refiner sample-local
preserves its serial arithmetic without disabling packing in the main DiT.
This is not a promise of bitwise equality for every packed operator or shape.

AdaLN timestep gathers retain the upstream dtype and backward reduction on both
paths. This feature does not introduce an FP32 gather correction into the shared
H3 implementation. Low-precision repeated-index reductions can vary even between
serial repeats, so packed-gradient comparisons must also measure that baseline
variability; a separate accumulation-precision fix must not silently change the
single-sample path. Sample-local refiner execution is restricted to cross-sample
packing; ordinary single-sample attention dispatch is preserved.

### Attention and validation

- `eager` / `sdpa`: explicit per-segment PyTorch SDPA reference path; main-DiT
  projections and MLPs still operate on compact cross-sample rows.
- `flash_attention_2` / `flash_attention_3` in `model.ops_implementation` resolve
  to VeOmni's local FA2/FA3 backends, and `flash_attention_2_hub` /
  `flash_attention_3_hub` to the Hugging Face Hub kernels
  (`kernels-community/flash-attn2` / `flash-attn3`, version 1). Each main-DiT
  layer uses one non-causal varlen call; refiner layers retain one call per
  sample. The kernel is loaded on the first multi-sample forward and needs
  BF16/FP16; unavailable kernels are not silently replaced with SDPA.

`tests/models/test_minimax_h3_packing.py` uses a native tiny model on CPU to
check packed-versus-serial outputs, losses and gradients (including mixed target
geometry and checkpoint recomputation), sample isolation, Ref2VA variable
references, forward-local SP padding and fail-closed inputs. SP collectives are
mocked there, so it does not establish distributed SP parity. The FA2/FA3 call
site is checked for all four backends with a kernel stub that asserts the
varlen layout; kernel correctness itself is covered by
`tests/ops/attention/flash/test_flash_attention.py`.
No end-to-end speedup or convergence is claimed. Benchmark against an equivalent,
tuned non-packed baseline before claiming a performance improvement.

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
        attn_implementation="sdpa",
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

## Prepared Ref2VA with CFG-aware LoRA

Use [minimax_h3_ref2va_cfg_offline.yaml](../../configs/dit/minimax_h3_ref2va_cfg_offline.yaml)
with a compatible Ref2VA checkpoint and prepared visual-reference caches.
It uses the existing `DiTTrainer`, native LoRA, FSDP2 and model-owned packing.
It does **not** add raw-reference encoding, audio-reference support or a
Training Adapter downloader. A training/de-distillation adapter is separate from
the trainable LoRA; this recipe does not load one implicitly.

```text
Prepared video/audio latents + paired conditions
                         |
            Shared timestep and noise
                /                 \
Negative layout                   Positive layout
(same targets and visual anchors)       |
        |                               |
Current DiT (no gradient)       Current DiT (gradient enabled)
                \                 /
                CFG-aware FM objective
                         |
          Existing trainer and optimizer
```

### Cache contract

The existing `dit_offline` transform deserializes **every column** using Python
pickle. Only load trusted caches. The following table describes the decoded
objects, not a new byte-storage schema or a direct Magnus-table reader.

| Field | Decoded value | Purpose |
| --- | --- | --- |
| `input_latents` | Float tensor `[1,24,T,H,W]` | Clean video target |
| `audio_input_latents` | Float tensor `[C,32,Ta]` | Clean audio target |
| `prompt_embeds` | Float tensor `[L,D]` | Conditional Qwen representation |
| `packed` | Dictionary from `build_packed_ref2va` | Conditional positions and segment bounds |
| `ref_visual_anchor` | Float tensor `[cond_rows,96]` | Prepared reference image/video rows |
| `has_audio` | Boolean | Set false for placeholder audio; its loss is zero |
| `unconditional_prompt_embeds` | Float tensor `[Lu,D]` | Required per sample when CFG calibration is enabled |
| `unconditional_packed` | Dictionary from the same layout builder | Required with the negative embedding; uses `Lu` and its own text token tags |

The negative condition is independently encoded for each sample. By default, it
retains the same reference visual context in the Qwen representation. It cannot derive those
embeddings by truncating or zeroing the conditional embedding. Both packed
layouts must have identical target/reference geometry, reference order and
compact `[text | cond | audio | video]` structure. Text lengths may differ.
Use the same source references and compatible encoder checkpoint for both caches;
structural checks cannot prove that two embeddings have the same semantic origin.

The default negative prompt is exactly `" "` (**one space**, not `""`), matching
native inference. Tokenization uses `add_special_tokens=False`, so an empty
string has no automatically inserted prompt tokens; the Ref2VA presentation
also rejects an empty string. A single space provides a nonempty placeholder
without a descriptive caption. This is an encoding convention, not a
mathematical requirement of CFG.

Use the same FL2VA/Ref2VA presentation, encoder checkpoint, resized keyframes or
prepared reference blocks as the conditional cache, including reference order
and video timestamps. This is the default negative policy and matches the
repository's optional two-pass inference CFG. It does not establish the original
model's distillation procedure or guarantee quality preservation.

#### Rationale and experimental scope

H3 is CFG-distilled: its conditional prediction already incorporates learned
guidance. CFG calibration uses a detached negative prediction from the current
model to transform the conditional prediction before applying flow-matching
supervision. The content of that negative condition therefore determines the
guidance direction being calibrated; it is not an arbitrary placeholder. This
training objective may mitigate loss of distilled guidance during fine-tuning,
but does not guarantee that it will prevent quality degradation.

Keeping the Qwen visual context makes the positive and negative conditions differ
primarily in their text prompt. This is the compatibility default because it
follows VeOmni's existing negative-condition construction. It is not evidence
that H3's original distillation used exactly this condition-dropping policy.
Likewise, single-pass inference from a distilled model does not reveal how its
teacher constructed negative conditions during training.

For controlled experiments, `drop_visual=True` generates a text-only Qwen
negative instead. It drops both the caption and Qwen-side visual information,
but **preserves DiT reference latent anchors**; it is not fully unconditional.
The contrast now includes differences in Qwen visual-semantic context as well
as text. This is a different guidance direction, not an equivalent or merely
cheaper implementation of the default. As a related community precedent,
[AI Toolkit's static-prompt encoder](https://github.com/ostris/ai-toolkit/blob/2fde764876d096f7bee8b5f5bd137b43aeee773a/extensions_built_in/sd_trainer/SDTrainer.py#L143-L151)
first encodes without per-sample control images, with a blank-control fallback
for models that require images. This motivates an experimental option, not a
claim that the two training implementations are equivalent.

Limited internal experiments observed better quality stability with
Qwen-visual-free negatives in some runs. Dataset coverage, training horizons,
and evaluation coverage were limited, so these observations are preliminary
and do not establish a generally superior policy. The option remains opt-in.
The policy is chosen when generating paired caches, not by overriding a
training flag on an existing dataset. Both policies share the same paired-cache
schema and CFG-calibrated loss, without a separate shared-embedding loader.

Further controlled experiments are planned with matched data, initialization,
and training hyperparameters. Evaluation will compare prompt adherence,
reference fidelity, temporal stability, and quality degradation across multiple
checkpoints. Results can inform future recommendations; these quality studies
are separate from distributed correctness smoke tests and are not prerequisites
for this implementation.

### Generate unconditional caches

The condition model's `add_unconditional_cache` helper reuses native inference's
Qwen presentation and encoding code. Run it while preparing the dataset,
using the already-loaded condition model in evaluation mode with its Qwen
encoder on the desired device. No DiT or VAE forward is needed for this call.
The new fields contain CPU tensors; the original sample and latent anchors are
preserved.

```python
# `condition_model` is the encoder-loaded MiniMaxH3ConditionModel used to
# produce the conditional caches (skip_encoder_load=False).
condition_model.eval()

# Per-sample Ref2VA cache. `sample` is the decoded CPU sample from the table
# above; refs are its original prepared reference blocks, in the same order.
# An image block contains kind="image", prepared_image=<resized PIL image>.
# A visual-video block contains kind="video", prepared_frames=<24 FPS frames>,
# ref_audio_t=0. Use the same preparation as for the positive cache.
sample_with_negative = condition_model.add_unconditional_cache(
    sample, negative_prompt=" ", ref_blocks=refs
)

# For FL2VA, instead supply the original resized keyframe images:
fl2va_with_negative = condition_model.add_unconditional_cache(
    fl2va_sample, negative_prompt=" ", keyframe_images=keyframe_images
)

# Experimental text-only Qwen negative, supported for both FL2VA and Ref2VA.
# No source images/frames are needed; the sample's DiT latent anchors remain.
text_only_negative = condition_model.add_unconditional_cache(
    sample, negative_prompt=" ", drop_visual=True
)
# Save each augmented sample using the existing OfflineEmbeddingSaver.save().
```

`add_unconditional_cache` writes `unconditional_prompt_embeds` and
`unconditional_packed`. It rebuilds the Qwen prefix length/tags and shifts the
target/reference positions consistently. It cannot recover original references
from a conditional embedding; keep the prepared source images/frames available
during dataset creation for the default policy. Reference audio is not supported
by this recipe. Neither policy infers negatives by truncating positive embeddings.

The text-only experiment has precedent in AI Toolkit's
[cached unconditional prompt construction](https://github.com/ostris/ai-toolkit/blob/2fde764876d096f7bee8b5f5bd137b43aeee773a/extensions_built_in/sd_trainer/SDTrainer.py).
Released H3 inference in
[AI Toolkit](https://github.com/ostris/ai-toolkit/blob/2fde764876d096f7bee8b5f5bd137b43aeee773a/extensions_built_in/diffusion_models/minimax_h3/src/pipeline.py)
and [vLLM-Omni](https://github.com/vllm-project/vllm-omni/blob/4c5541cfc17143f80bdb89bbb7a5840b08bb52c6/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py)
uses the CFG-distilled single-pass path. This absence of an inference negative
branch does **not** identify the negative-conditioning policy of the original
distillation teacher.

### CFG-calibrated FM and loss settings

Let `p` and `u` be conditional and detached unconditional FM velocities from the
**same current model**, including its trainable LoRA, and `y = noise - clean`.
The negative pass runs first under `no_grad`; it is not a frozen-base teacher.

There is one objective, with a configurable curvature exponent `k`:

```text
p_calibrated = (p + (s - 1) * stopgrad(u)) / s
loss = s ** (2 - k) * MSE(p_calibrated, y)
```

| `training_cfg_curvature_power` | Relative conditional curvature | At `s=4` |
| --- | --- | --- |
| `2.0` (default) | `1/s²` | `1/16` |
| `1.0` | `1/s` | `1/4` |
| `0.0` | `1` | `1` |

Any finite value in `[0, 2]` is supported. CFG-calibrated FM is a descriptive
name for the existing CFG-aware transformation, not a novel algorithm: `k=2`
is the inverse-CFG formula and `k=0` is algebraically equivalent to
`MSE(p, s*y - (s-1)*stopgrad(u))` (the target-guidance form).
The optimum is unchanged for fixed `u` and `s`, but the gradient scale changes.
With sigma-dependent scales this also reweights different samples/modalities;
it cannot in general be replaced by a single learning-rate change. Loss values
are not directly comparable across curvature settings, and this objective does
not guarantee preservation of generation quality.

This objective distills guidance into the conditional prediction. At inference,
the default `cfg_scale=1` uses that prediction directly; applying additional
inference-time CFG can compound the guidance and should be validated separately.

`sigma = 1 - t` is the effective noise level from the existing scheduler. With
`training_cfg_schedule: sigma`, video and audio use their respective sigmas:
`s = 1 + (training_cfg_scale - 1) * sigma`. For a configured scale of `4`,
`sigma=0.1` gives `s=1.3`, while `sigma=0.9` gives `s=3.7`. This concentrates
stronger CFG calibration at high noise and leaves low-noise examples closer to
ordinary FM, giving them more room to learn from new data. `constant` instead
applies the configured scale at every noise level, equally to video and audio,
and is the default in the example recipe.

The existing shifts strongly favor high sigma. Enumerating all 1,000 scheduler
indices at scale `4`, before model-dtype rounding, gives:

| Modality / shift | Mean scale | Median scale | Scale > 3 | Scale < 2 |
| --- | ---: | ---: | ---: | ---: |
| Video / 12 | 3.53 | 3.77 | 85.8% | 3.9% |
| Audio / 3 | 3.03 | 3.25 | 60.0% | 14.2% |

Therefore sigma scheduling gives relatively little low-noise relief for video
with the shipped shifts. Separate modality scales are an **opt-in training
heuristic** based on their effective corruption levels, not a reconstruction
of a known original distillation scale. Native inference applies a single CFG
scale to both modalities. A quality ablation is needed to establish whether
per-modality scheduling helps; the formula alone does not show this. Neither
schedule changes the noise sampling distribution or guarantees quality.

| Condition config key | Default | Meaning |
| --- | --- | --- |
| `training_cfg_scale` | `1.0` | Scale one disables the extra branch and preserves upstream RNG/loss behavior |
| `training_cfg_schedule` | `constant` | `constant` uses the configured scale everywhere; `sigma` varies it with each modality's effective noise level |
| `training_cfg_curvature_power` | `2.0` | Curvature exponent `k` in `[0,2]`; relative curvature `1/s**k` |

Noise sampling is unchanged from upstream: a uniform scheduler index is shared
by video and audio, with their respective existing shifts. Existing scheduler
weights are retained; no additional sigma-bin weights are applied.
No mixed raw-FM objective, stochastic CFG gate, step warmup or drift anchor is
silently enabled by this configuration.

### Validation boundary

`add_unconditional_cache` runs full value validation on CPU before returning
the sample: negative embedding finiteness, compact positions, modality tags
and matching target/reference coordinates. Custom cache producers must call
`validate_unconditional(positive_packed, negative_packed, negative_embedding,
positive_embedding)` from `minimax_h3_core.cfg_conditioning` before saving each
sample. Values are not re-scanned every epoch; caches must remain unchanged
after validation. Training does not filter or repair NaN embeddings.

At training time, H3 checks only tensor shapes, modality row counts and scalar
geometry. These metadata checks do not read tensor values and work after the
ordinary device transfer. Remapping reuses the positive branch's noise,
reference rows and unique timesteps. No validation markers or shared
`DiTTrainer` changes are required.

CPU tiny-model tests cover the objective gradients, paired noise, reference
geometry, cached embeddings, checkpoint recomputation and packed/serial loss
equivalence. The accelerator sync test covers CFG disabled and paired CFG inputs;
running that test requires a GPU. Existing H3 packing restrictions remain, including no multi-sample
checkpoint offload. The [four-GPU smoke test report](minimax_h3_smoke_test.md)
records forward loss parity and parameter updates for SP1/SP2, together with
an unresolved SP2 raw-gradient scaling discrepancy. It is not a full numerical
equivalence or production-quality validation. Pretrained quality and throughput
still require dedicated evaluation.
