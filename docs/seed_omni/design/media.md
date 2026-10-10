# SeedOmni Media: Video and Audio

This page describes how SeedOmni reads and writes image, video and audio items,
the timeline metadata they carry, and the decided design for audio-bearing video
(video with its own sound track), which is not implemented yet.

| Capability | Status |
|------------|--------|
| Reading standalone audio, silent video and video with sound (training and inference) | implemented |
| Silent video understanding (see [Qwen3-VL](../models/qwen3vl.md)) | implemented |
| Saving generated video with its sound track muxed in | implemented |
| Understanding audio-bearing video (time interleave + TMRoPE) | design only (§5) |

## 1. Carriers

Media travels as ordinary `ConversationItem`s (see
[Architecture](architecture.md#22-conversationitem-the-data-carrier-utilsconversationpy)):

- `type="image"`: the decoded image.
- `type="audio"`: a bare waveform, for speech input and output. A standalone sound
  item needs no shared timeline.
- `type="video"`: `value` is a `VideoInputs` (`veomni/data/seed_omni/utils/video.py`)
  holding the frames and an optional `audio` track; `VideoInputs.has_audio` tells
  whether the clip has sound.

Timelines are **not** in the payload. They live on the item's `meta` under
`video_metadata` / `audio_metadata` (source fps, source frame count, the kept
frames' `frames_indices`, the audio sample rate; see
`veomni/data/seed_omni/utils/media_metadata.py`). The payload is transient
(encoders overwrite `value` with embeddings) and `meta` is the only channel that
survives the item's lifecycle, so anything a module derives about timing is
written back to `meta` as well. An in-video sound track states its rate under the
same `audio_metadata` key a standalone audio item uses, so consumers read one
spelling regardless of how the sound arrived.

## 2. Reading

Both callers that turn media references into conversation items, the training
transform and inference request building, go through one table:
`MEDIA_FETCHERS` in `veomni/data/seed_omni/utils/media.py`. Every fetcher returns
`[(payload, meta), ...]`; the payload becomes `item.value` and the meta is merged
into `item.meta`. Adding a modality is one entry there and serves training and
inference at once, so "the same clip" means the same thing wherever it is read
(for example, an inference audio item always carries its sampling rate, which
its consumers need and cannot recover from the samples).

- Sound belonging to a clip is **one** video entry plus `use_audio_in_video`,
  never a video paired with a separate audio entry: the two streams share one
  timeline, and splitting them across items throws the alignment away.
- One flat set of decode options (`fps`, `max_frames`, `frames_fps`,
  `image_max_pixels`, `video_max_pixels`, `audio_sampling_rate`,
  `use_audio_in_video`) is offered to every fetcher; each ignores what it does not
  read, so one caller-side config serves all modalities.
- References for a modality with no fetcher are refused by name, and so is a
  video reference when the decode stack is missing, so a missing system
  dependency never turns into a text-only answer to a question about a video.

A consumer reads the tracks it can use and skips the rest: a vision tower takes
`VideoInputs.video` and ignores `.audio`. That is correct for a model that cannot
hear (Qwen3-VL). For a model that could, `use_audio_in_video=True` currently
trains and answers on the frames alone, because no backbone implements the
interleave in §5 yet; decode with `use_audio_in_video=False` until it does,
rather than paying to decode a track nothing reads.

## 3. Timeline rules

These rules apply to every module that consumes or produces timed media.

- **`fps` is always a real rate.** `VideoMetadata` subclasses
  `transformers.video_utils.VideoMetadata`, so an HF video processor takes it as
  is, and it refuses construction without a positive finite `fps`. That keeps
  HF's fallbacks (`sampled_fps` returning 24, the Qwen3-VL processor writing
  `metadata.fps = 24` back) from firing, since they only trigger on a missing
  rate. Every source supplies one: a container states it (one that does not is
  refused), a pre-decoded frame list is declared through `mm_configs.frames_fps`
  (no default; an undeclared list is refused), and a generated clip is written at
  one. `frames_indices` defaults to the full range and `duration` to
  `total_num_frames / fps`.
- **The metadata does the time arithmetic, not its callers.** Ask
  `VideoMetadata` for a time, never for a rate to divide by:
  `frame_timestamps(temporal_patch_size)` for per-frame times,
  `seconds_per_patch(temporal_patch_size)` for the scalar TMRoPE scales its
  temporal row by, `duration` (`duration_seconds` on `AudioMetadata`) for a
  clip's span. Never divide `frames_indices` (source-frame numbers) by a *target*
  sampling rate: frame 300 of a 30 fps clip is at 10 s, but 300 / 2 reads as
  150 s (the bug fixed in
  [#1165](https://github.com/ByteDance-Seed/VeOmni/pull/1165)). For the same
  reason `requested_fps` is recorded but never used as a divisor, and
  `second_per_grid` must not be derived as `temporal_patch_size / requested_fps`:
  a 25 fps source trimmed to "2 fps" keeps frames 0.48 s apart, not 0.5 s.
- **Nothing re-samples an already-sampled clip.** `load_video` picks the frames;
  a downstream processor is called with `do_sample_frames=False` and handed the
  source axis verbatim (today `qwen3vl/vision`; every later video consumer owes
  the same). A second sampling pass re-picks frames, after which
  `frames_indices` describes frames the vision tower never saw. Padding the tail
  up to a whole `temporal_patch_size` is the one thing a processor may still do,
  and `frame_timestamps` pads to match.
- Both streams are measured from the container's zero, so comparing the two
  durations is how a caller detects sound that stops before the frames do.

## 4. Writing generated media

A module that produces a clip emits the same carrier it would receive: one
`type="video"` item whose `value` is a `VideoInputs` (sound present means the clip
has sound) with both timelines on `meta`. `OmniInferencer._save_generated_media`
writes one file per item through `save_video`, muxing the sound track into the
container rather than leaving an `.mp4` and a `.wav` side by side.

Every saver has one call shape, `save(path, value, meta)`, and is looked up in
`MEDIA_SAVERS` (`veomni/data/seed_omni/utils/media.py`, keyed like
`MEDIA_FETCHERS`). `meta` is the item's whole `ConversationItem.meta`; each saver
reads what its modality needs, so the caller knows nothing per modality and a new
modality is one entry in each table.

Both rates come from `meta` and neither has a default: `VideoMetadata.fps` for the
container and `AudioMetadata.sampling_rate` for the sound track, each refused when
absent. A wrong rate is not a loud failure but a file that plays at the wrong
speed or drifts out of lip sync, which nothing downstream can tell apart from a
bad generation. A generated clip needs no second metadata type: it is its own
source, so `fps` is both the source and the playback rate and `frames_indices` is
the full range, which keeps `frame_timestamps` meaningful for it.

## 5. Audio-bearing video understanding (design)

### Reference: transformers `qwen2_5_omni` / `qwen3_omni_moe`

- **Messages.** The user writes a single `video` turn and no separate audio item.
  `process_mm_info(..., use_audio_in_video=True)` pulls the audio out of the
  container into its own list. The chat template renders only
  `<|vision_bos|><|VIDEO|><|vision_eos|>`; `use_audio_in_video` is a processor
  flag, not a template token.
- **Tokens.** `processor.replace_multimodal_special_tokens` expands that one
  `<|VIDEO|>` block into a time-interleaved run
  `<|vision_bos|><|audio_bos|> ...(interleaved VIDEO/AUDIO placeholders)... <|audio_eos|><|vision_eos|>`.
  Qwen2.5-Omni interleaves on fixed time chunks (`seconds_per_chunk`, default
  2 s); Qwen3-Omni merge-sorts per-token timestamps. Time indices come from
  `video_second_per_grid` / `position_id_per_seconds`, so both streams land on one
  timeline, which is what TMRoPE (time-aligned multimodal RoPE) is built on.
- **Encoders.** The processor emits two tensor streams from the same file: video
  (`pixel_values_videos` + `video_grid_thw`) and audio (`input_features` +
  `feature_attention_mask`). The two encoders run independently and the
  interleaved placeholders are back-filled by `masked_scatter`; because the
  placeholders were laid out in time order, the scatter restores alignment.

### SeedOmni design

- **Carrier:** one `type="video"` item carrying both streams (§1), with both
  timelines on `meta`.
- **Text encoder (layout only):** choose the outer wrapping from
  `VideoInputs.has_audio`: plain video gets `<|vision_bos|> ... <|vision_eos|>`,
  audio-bearing video gets `<|vision_bos|><|audio_bos|> ... <|audio_eos|><|vision_eos|>`.
  It does **not** interleave, and keeps the compressed loss layout (an AV item
  contributes a single `-100` row at decode), so it never needs the exact
  placeholder counts.
- **Video module:** `item.value.video` to video embeddings.
- **Audio module** (Whisper-style mel features and audio encoder):
  `item.value.audio` to audio embeddings.
- **Backbone:** owns the time interleave. It merges the item's video and audio
  embeddings into the flat `inputs_embeds` by timestamp and builds the aligned
  TMRoPE position ids in the same pass; the span's total length and labels are its
  responsibility too.

This split keeps the data model-agnostic (one media item) and both encoders pure
embedding providers, so the two coupled concerns, interleave order and
time-aligned positions, sit in the backbone, which already owns splicing and
position construction.
