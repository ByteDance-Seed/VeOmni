# SeedOmni Media: Video and Audio

This page describes how SeedOmni reads and writes image, video and audio items,
the timeline metadata they carry, and the decided design for audio-bearing video
(video with its own sound track), which is not implemented yet.

| Capability | Status |
|------------|--------|
| Reading standalone audio, silent video and video with sound (training and inference) | implemented |
| Silent video understanding (e.g. Qwen3-VL) | implemented |
| Saving generated video with its sound track muxed in | implemented |
| Understanding audio-bearing video (time interleave + TMRoPE) | design only (§5) |

## 1. Carriers

Media travels as ordinary `ConversationItem`s (see
[Architecture](architecture.md#22-conversationitem-the-data-carrier-utilsconversationpy)):

- `type="image"`: the decoded image.
- `type="audio"`: a bare waveform for standalone speech input and output.
- `type="video"`: `value` is a `VideoInputs` (`veomni/data/seed_omni/utils/video.py`)
  holding the frames and an optional `audio` track (`VideoInputs.has_audio`).

Timelines live on `item.meta` under `video_metadata` / `audio_metadata` (source
fps, source frame count, kept `frames_indices`, and audio sample rate; see
`veomni/data/seed_omni/utils/media_metadata.py`) because encoders overwrite
`item.value` with embeddings while `meta` persists across the graph. Standalone
audio and in-video audio share the `audio_metadata` key so consumers read one
spelling.

## 2. Reading

Both training transforms and inference request builders turn media references
into conversation items through `MEDIA_FETCHERS` in
`veomni/data/seed_omni/utils/media.py`. Every fetcher returns `[(payload, meta),
...]`; the payload becomes `item.value` and `meta` is merged into `item.meta`,
so training and inference produce identical payloads and metadata.

- Represent a clip with sound as a single `video` entry with
  `use_audio_in_video=True` so the video frames and audio track share one
  timeline.
- Pass decode options (`fps`, `max_frames`, `frames_fps`, `image_max_pixels`,
  `video_max_pixels`, `audio_sampling_rate`, `use_audio_in_video`) in one flat
  `mm_configs` dict; each fetcher reads the keys that apply to its modality.
- A consumer reads the tracks it uses from `VideoInputs` (for example, a
  vision-only tower reads `VideoInputs.video`).

## 3. Timeline rules

- **`fps` is always a real rate.** `VideoMetadata` subclasses
  `transformers.video_utils.VideoMetadata` and requires a positive finite `fps`
  from the container, from `mm_configs.frames_fps` (for pre-decoded frame
  lists), or from the generator. `frames_indices` defaults to the full range and
  `duration` to `total_num_frames / fps`.
- **Use `VideoMetadata` methods for time arithmetic.** Because `frames_indices`
  stores source-frame numbers on the source `fps` axis, query
  `frame_timestamps(temporal_patch_size)` for per-frame times,
  `seconds_per_patch(temporal_patch_size)` for TMRoPE temporal scaling, and
  `duration` (`duration_seconds` on `AudioMetadata`) for clip span.
- **Sample frames once in `load_video`.** Pass `do_sample_frames=False` to
  downstream video processors so the vision tower sees the exact frames recorded
  in `frames_indices` (`frame_timestamps` pads the tail to match
  `temporal_patch_size`).
- Both streams are measured from container time zero, so their durations can be
  compared directly.

## 4. Writing generated media

A module that produces a clip emits a `type="video"` item whose `value` is a
`VideoInputs` with both timelines on `meta`.
`OmniInferencer._save_generated_media` writes one file per item through
`MEDIA_SAVERS` (`veomni/data/seed_omni/utils/media.py`, signature `save(path,
value, meta)`), muxing the video frames and optional audio track into a single
container using `VideoMetadata.fps` and `AudioMetadata.sampling_rate` from
`item.meta`.

## 5. Audio-bearing video understanding (design)

### Reference: transformers `qwen2_5_omni` / `qwen3_omni_moe`

- **Messages.** A single `video` turn with `use_audio_in_video=True` extracts the
  container audio and renders `<|vision_bos|><|VIDEO|><|vision_eos|>` in the
  chat template.
- **Tokens.** `processor.replace_multimodal_special_tokens` expands `<|VIDEO|>`
  into a time-interleaved `<|vision_bos|><|audio_bos|> ...(VIDEO/AUDIO
  placeholders)... <|audio_eos|><|vision_eos|>` sequence using
  `video_second_per_grid` / `position_id_per_seconds` for TMRoPE alignment.
- **Encoders.** The video and audio encoders run independently on their
  respective streams and scatter embeddings into the time-ordered placeholders.

### SeedOmni design

- **Carrier:** one `type="video"` item carrying both streams (§1) and their
  timelines on `meta`.
- **Text encoder (layout):** wraps the item according to
  `VideoInputs.has_audio` (`<|vision_bos|> ... <|vision_eos|>` for silent video,
  `<|vision_bos|><|audio_bos|> ... <|audio_eos|><|vision_eos|>` for
  audio-bearing video) and keeps the compressed loss layout.
- **Video module:** encodes `item.value.video` into video embeddings.
- **Audio module:** encodes `item.value.audio` into audio embeddings.
- **Backbone:** merges the item's video and audio embeddings into `inputs_embeds`
  by timestamp and constructs the aligned TMRoPE position ids and span labels in
  the same pass.
