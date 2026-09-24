# Design Note: Video With Audio (Qwen-Omni style av-video)

> **Status: understanding side not implemented (design-only).** Audio-bearing
> video *understanding* is still unsupported; silent video understanding **is**
> implemented (see [`example_models/qwen3vl.md`](example_models/qwen3vl.md)). The
> data layer already decodes and carries the audio stream (`VideoInputs.audio` in
> `veomni/data/seed_omni/utils/video.py`), but no encoder or backbone consumes it
> — none of the interleave / TMRoPE work below exists. This note records the
> intended design so a future implementation has a decided starting point.
>
> The **generation side does** consume it: `save_video` writes a generated clip
> and muxes its sound track into one container (§ *Generated clips*). Writing a
> file needs no interleave or shared position ids, which is why that half could
> land first.
>
> What *does* exist now is the standalone `type="audio"` conversation item (a
> bare waveform, no video), added for Qwen3-Omni speech in/out. That is a
> different carrier shape on purpose: one sound item on its own needs no shared
> timeline, so it needs none of the interleave or TMRoPE machinery below.
>
> **Reading is done for all three shapes** (standalone audio, silent video,
> video with sound), on training *and* inference, through one decode table —
> `MEDIA_FETCHERS` in `veomni/data/seed_omni/utils/media.py` (§ *Reading*). What
> is missing for av is only the *consumption*: a backbone that interleaves the
> two streams. A model can therefore be handed an audio-bearing clip today and
> still has to implement the interleave below to read it.
>
> A consumer reads the tracks it can use and skips the rest: a vision tower
> takes `VideoInputs.video` and ignores `.audio`, so an audio-bearing clip is not
> an error for a model without the interleave. That is the right behaviour for a
> model that cannot hear at all (Qwen3-VL). For one that could (Qwen3-Omni), it
> means `use_audio_in_video=True` currently trains and answers on the frames
> alone, where upstream transformers would interleave the sound in — decode with
> `use_audio_in_video=False` until the interleave lands, rather than paying to
> decode a track nothing reads.

## Reference implementation (transformers `qwen2_5_omni` / `qwen3_omni_moe`)

- **Message level**: the user writes a single `video` turn and no separate audio
  item. The audio is pulled out of the mp4 by
  `process_mm_info(..., use_audio_in_video=True)` and placed in its own `audio`
  list. The chat template only renders
  `<|vision_bos|><|VIDEO|><|vision_eos|>` — `use_audio_in_video` is a
  **processor flag**, not a template token.
- **Token level**: `processor.replace_multimodal_special_tokens` expands that one
  `<|VIDEO|>` block into a **time-interleaved** run of
  `<|vision_bos|><|audio_bos|> …(interleaved VIDEO/AUDIO placeholders)… <|audio_eos|><|vision_eos|>`.
  - Qwen2.5-Omni interleaves on fixed time chunks (`seconds_per_chunk`, default 2s).
  - Qwen3-Omni merge-sorts per-token timestamps (whichever stream's next token is
    earlier goes first).
  - Time indices come from `video_second_per_grid` / `position_id_per_seconds`, so
    video and audio land on one shared timeline — this is also what **TMRoPE**
    (time-aligned multimodal RoPE) is built on.
- **Split across two encoders**: the split happens in the processor, which emits
  two tensor streams from the same mp4 — video (`pixel_values_videos` +
  `video_grid_thw`) and audio (`input_features` + `feature_attention_mask`). In
  the model, the two encoders encode independently and the interleaved
  placeholders are back-filled separately by `masked_scatter` (VIDEO slots ← video
  embeds, AUDIO slots ← audio embeds). Because the placeholders were already laid
  out in time order, the scatter restores time alignment.

## SeedOmni target design (decided)

- **Carrier**: one `conversation_list` item with `type="video"` and
  `value = video_inputs`, carrying both streams as payload —
  `VideoInputs.audio` is optional (present ⇒ the clip has sound).
  The two timelines the interleave and TMRoPE need are **not** in that payload.
  They sit on the item's `meta` under `video_metadata` / `audio_metadata`
  (source fps, source frame count, the kept frames' `frames_indices`, the audio
  sample rate; see `veomni/data/seed_omni/utils/media_metadata.py`), because
  the payload is transient — the encoders overwrite `value` with embeddings and
  `meta` is the only channel that survives the item's lifecycle. Anything a
  module *derives* still has to be written back the same way — the
  `qwen3omni_vision` module (landing on the `omniv2.qwen3omni` branch) does
  that for `second_per_grid`.
  An in-video track states its rate under the same `audio_metadata` key a
  standalone `type="audio"` item uses, so the interleave reads one spelling
  regardless of how the sound arrived.
- **The metadata does the time arithmetic, not its callers.** This is the rule
  for the modules below, which are still being built — today the data layer
  writes the timeline and the accessors exist for them to read. Ask
  `VideoMetadata` for a time, never for a rate to divide by:
  `frame_timestamps(temporal_patch_size)` for per-frame times,
  `seconds_per_patch(temporal_patch_size)` for the scalar TMRoPE scales its
  temporal row by, `duration` (`duration_seconds` on `AudioMetadata`) for a
  clip's span.
  The one thing that must never happen is dividing `frames_indices`
  (source-frame numbers) by a *target* sampling rate, which is the bug
  [#1165](https://github.com/ByteDance-Seed/VeOmni/pull/1165) fixed on the
  BaseTrainer path — frame 300 of a 30 fps clip is 10 s, but 300/2 reads as
  150 s. So there is no bare kept-frame-rate field to pair it with, and
  `requested_fps` is recorded but is never a divisor. Note this covers
  `second_per_grid` too: upstream derives it as
  `temporal_patch_size / requested_fps`, which is the same axis error — a
  25 fps source trimmed to "2 fps" keeps frames 0.48 s apart, not 0.5 s, so
  every position id in and after the clip shifts.
  Both streams are measured from the container's zero, so comparing the two
  durations is how a caller notices sound that stops before the frames do.
- **`fps` is always a real rate.** `VideoMetadata` subclasses
  `transformers.video_utils.VideoMetadata`, so an HF video processor takes it
  as is, and it refuses construction without a positive finite `fps`. That is
  what makes the inheritance safe: HF's fallbacks (`sampled_fps` returning 24,
  the Qwen3-VL processor writing `metadata.fps = 24` back) only fire on a
  missing rate. Every source has to supply one — a container states it (one that
  doesn't is refused), a pre-decoded frame list is declared via
  `mm_configs.frames_fps` (no default; an undeclared list is refused), and a
  generated clip is written at one. `frames_indices` defaults to the full range
  and `duration` to `total_num_frames / fps`.
- **Nothing re-samples an already-sampled clip.** `load_video` picks the frames;
  a downstream processor is called with `do_sample_frames=False` and handed the
  source axis verbatim (today that is `qwen3vl/vision`, the one video-consuming
  module; every later one owes the same). A second sampling pass re-picks frames, after
  which `frames_indices` describes a set the vision tower never saw — the other
  half of #1165. It bites even when the rates look equal: a 25 fps source
  trimmed to "2 fps" actually lands at 2.083 fps, enough for the processor to
  drop a frame. Tail padding up to a whole `temporal_patch_size` is the one
  thing it may still do, and `frame_timestamps` pads to match.
- **`text_encoder` (layout only)**: choose the outer wrapping from whether
  `VideoInputs.has_audio` — plain video gets
  `<|vision_bos|> … <|vision_eos|>`, audio-bearing video gets
  `<|vision_bos|><|audio_bos|> … <|audio_eos|><|vision_eos|>` (audio inside,
  vision outside). It does **not** interleave. It keeps the Janus-style compressed
  loss (an av item contributes a single `-100` row at decode), so the text encoder
  never needs to pre-expand the exact video/audio placeholder counts.
- **video module**: `item.value.video` (frames) → video embeds.
- **audio module** (a new modality — Whisper-style mel features + audio encoder):
  `item.value.audio` → audio embeds.
- **llm backbone**: owns the time interleave — merges the item's video embeds and
  audio embeds into the flat `inputs_embeds` by timestamp, and builds the aligned
  **TMRoPE** position ids in the same pass. The total length of the av span and
  its labels are the backbone's responsibility too.

The point of this split: the data stays model-agnostic (one media item) and both
encoders stay pure embed providers, so the only two coupled concerns (interleave
order and time-aligned positions) are concentrated in the backbone, which already
owns splice and position construction.

## Reading (implemented)

Both callers that turn refs into conversation items — the training transform and
inference request building — go through one table, `MEDIA_FETCHERS` in
`veomni/data/seed_omni/utils/media.py`. Every fetcher returns
`[(payload, meta), ...]`: the payload becomes `item.value`, the meta is merged
into `item.meta`. Adding a modality is one entry there and serves both sides at
once.

Sharing it is what makes "the same clip" mean the same thing wherever it is read.
Request building used to have its own image-only loader that attached no
metadata, which had two consequences worth naming because they are the kind a
new model inherits silently:

- an audio item built for inference carried **no sampling rate**, a fact its
  consumers require (a 16 kHz tower, a 24 kHz codec each resample for
  themselves) and cannot recover from the samples;
- `videos` raised `NotImplementedError` at the request layer while the data layer
  had been decoding clips for training all along.

Sound belonging to a clip is **one** video entry plus `use_audio_in_video`, never
a video paired with a separate audio entry: the two streams share the timeline
the interleave above is built on, and splitting them across two items throws away
the alignment that made them one clip. One flat bag of decode knobs (`fps` /
`max_frames` / `frames_fps` / `image_max_pixels` / `video_max_pixels` /
`audio_sampling_rate` / `use_audio_in_video`) is offered to every fetcher, each dropping what it does not
read, so one caller-side config serves all modalities.

Refs for a modality with no fetcher are refused by name, and so is a video ref
when the decode stack is absent — the empty list that used to come back turned a
missing system dependency into a text-only answer to a question about a video.

## Generated clips (implemented)

A module that *produces* a clip emits the same carrier it would receive — one
`type="video"` item whose `value` is a `VideoInputs` (sound present ⇒ the clip has
sound) with both timelines on `meta`. `OmniInferencer._save_generated_media` then
writes one file per item through `save_video`, muxing the sound track in rather
than dropping an `.mp4` and a stray `.wav` beside each other.

Every saver has one call shape, `save(path, value, meta)`, and is looked up in
`MEDIA_SAVERS` (`veomni/data/seed_omni/utils/media.py`, keyed like
`MEDIA_FETCHERS`). `meta` is the item's whole `ConversationItem.meta`, not fields
the caller picked out: each saver reads what its modality needs, so the caller
knows nothing per modality and a new modality is one entry in each table.

Both rates come off the item's `meta`, and neither has a default:
`VideoMetadata.fps` for the container and `AudioMetadata.sampling_rate` for the
sound track, each refused when absent — the rule `save_audio` already set, for the
same reason. A wrong rate on either stream is not a loud failure but a file that
plays at the wrong speed or drifts out of lip sync, which nothing downstream can
tell apart from a model that generated it badly. `write_video_audio` would happily
infer a missing audio rate from `samples / duration`, so the refusal is what keeps
that guess out.

A generated clip needs no second metadata type: it *is* its own source, so nothing
was sub-sampled, `fps` is both the source and the playback rate, and
`frames_indices` is the full range — which keeps `frame_timestamps` meaningful for
it. A producer reads its own rates off the model: MiniMax-H3, the first such
producer heading here, exposes `MiniMaxH3Pipeline.frame_rate` beside
`audio_vae.sample_rate`.
