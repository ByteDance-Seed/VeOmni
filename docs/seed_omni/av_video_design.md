# Design Note: Video With Audio (Qwen-Omni style av-video)

> **Status: not implemented (design-only).** SeedOmni V2 has no audio modality,
> so audio-bearing video is unsupported — silent video understanding *is*
> implemented (see [`example_models/qwen3vl.md`](example_models/qwen3vl.md)). The
> data layer already decodes and carries the audio stream (`VideoInputs.audio` in
> `veomni/data/seed_omni/video_utils.py`), but nothing downstream consumes it.
> This note records the intended design so a future implementation has a decided
> starting point.

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

## SeedOmni V2 target design (decided)

- **Carrier**: one `conversation_list` item with `type="video"` and
  `value = video_inputs`; `meta["audio_stream"]` is optional (present ⇒ the clip
  has sound). The encoder must also write the timeline metadata into `meta`
  (video's `second_per_grid` / fps, audio's frame rate), otherwise the backbone
  cannot compute the interleave order or TMRoPE.
- **`text_encoder` (layout only)**: choose the outer wrapping from whether
  `meta["audio_stream"]` exists — plain video gets
  `<|vision_bos|> … <|vision_eos|>`, audio-bearing video gets
  `<|vision_bos|><|audio_bos|> … <|audio_eos|><|vision_eos|>` (audio inside,
  vision outside). It does **not** interleave. It keeps the Janus-style compressed
  loss (an av item contributes a single `-100` row at decode), so the text encoder
  never needs to pre-expand the exact video/audio placeholder counts.
- **video module**: `item.value` (frames) → video embeds.
- **audio module** (a new modality — Whisper-style mel features + audio encoder):
  `item.meta["audio_stream"]` → audio embeds.
- **llm backbone**: owns the time interleave — merges the item's video embeds and
  audio embeds into the flat `inputs_embeds` by timestamp, and builds the aligned
  **TMRoPE** position ids in the same pass. The total length of the av span and
  its labels are the backbone's responsibility too.

The point of this split: the data stays model-agnostic (one media item) and both
encoders stay pure embed providers, so the only two coupled concerns (interleave
order and time-aligned positions) are concentrated in the backbone, which already
owns splice and position construction.
