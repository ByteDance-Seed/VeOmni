# SeedOmni Data Format

This guide describes the **on-disk conversation schema** used by SeedOmni
(`data_type: seedomni`) and the multisource preprocessors under
``veomni/data/seed_omni/preprocess.py``. Preprocessors normalize each dataset's
raw `conversations` column into a uniform `user` / `assistant` turn list so the
same pipeline feeds understanding (I2T), generation (T2I), and interleaved
editing / UG training.

## Pipeline overview

```
parquet row (conversations + images)
        │
        ▼  conv_preprocess(<source_name>, …)
        │  veomni/data/seed_omni/preprocess.py
        ▼
seedomni_transform.process_seedomni_example
        │  veomni/data/seed_omni/seedomni_transform.py
        ▼
raw_batch["conversation_list"]  →  OmniModel modules (SigLIP / VQVAE / …)
```

Set `data.data_type: seedomni` and point `data.train_path` at a multisource YAML
(see ``configs/seed_omni/Janus/data.yaml``). Each source's ``names`` entry
must match a key in ``SEED_OMNI_PREPROCESSOR_REGISTRY`` (or add your own
preprocessor in ``veomni/data/seed_omni/preprocess.py``).

## On-disk row schema (parquet / jsonl)

Each training sample is one row:

| Column | Type | Description |
|--------|------|-------------|
| `source_name` | `str` | Preprocessor id, e.g. `imagenet1k`, `tulu-3-sft-mixture`, `sharegpt4v_cap_100k` |
| `conversations` | JSON bytes / list | Chat messages (see below) |
| `images` | `list[bytes]` | PNG/JPEG bytes, consumed **in conversation order** for every `{"type": "image"}` placeholder |
| `videos` | `list[bytes \| str]` | Optional; paired in order with every `{"type": "video"}` placeholder |
| `audios` | `list[bytes \| str]` | Optional; paired in order with every `{"type": "audio"}` placeholder |

Media bytes are **not** embedded inside `conversations`. Each media item is a
placeholder; `seedomni_transform` pairs them with `images[0]`, `images[1]`, …
in order, and likewise per modality.

- **Audio** is decoded at the clip's native sampling rate and recorded in
  `meta["audio_metadata"].sampling_rate` (both for standalone audio items and
  in-video audio tracks), so downstream modules can resample to their own target
  rates (e.g. 16 kHz for an audio tower, 24 kHz for a speech codec) while
  preserving wall-clock alignment for TMRoPE. `wav` / `flac` / `ogg` / `mp3`
  decode directly; `m4a` / `aac` / `wma` fall back to `audioread`. For raw
  waveform arrays without a container header, declare their native rate via
  `data.mm_configs.audio_sampling_rate`. Segment audio clips to fit the encoder
  window (e.g. ≤ 30 s for the Qwen3-Omni tower).
- **Video** is decoded and frame-sampled once in the data layer, recording the
  source clip's `fps`, frame count, and kept `frames_indices` in
  `meta["video_metadata"]` (query `frame_timestamps()` and `duration` on the
  metadata; see [Media](../design/media.md#3-timeline-rules)). For pre-decoded
  frame lists (PIL images or image bytes), declare the source capture rate via
  `data.mm_configs.frames_fps`.

## Normalized turn layout

Each preprocessor in `veomni/data/seed_omni/preprocess.py` converts a source row
into `(constructed, media_refs)`:

```python
constructed = [
    ["user", ("image", None, {"_img_tag": "und"}), ("text", "Describe this image in detail.")],
    ["assistant", ("text", "A cat on a sofa.")],
]
media_refs = {"image": ["<user_png_bytes>"]}
```

Rules:

- **Content types**: `text`, `image`, `video`, `audio`.
- **Text** carries the string in its tuple value `("text", "...")`.
- **Media** uses `None` as its tuple placeholder `("image", None[, meta])` and
  is paired positionally with `media_refs[type]` (`images`, `videos`, `audios`).
- **Input vs target** is determined by **`role`** (and optional `meta["_img_tag"]`):
  - `role == "user"` + `type == "image"` (`_img_tag: "und"` / `"edit"`) → SigLIP / context encoder input
  - `role == "assistant"` + `type == "image"` (`_img_tag: "gen"`) → VQVAE / generation target
  - `role == "user"` + `type == "audio"` → audio-encoder input
  - `role == "assistant"` + `type == "audio"` → target speech for a talker
    module.

## Built-in source schemas

### 1. Understanding (I2T — `sharegpt4v_cap_100k` / `sharegpt4v_sft`)

ShareGPT-style `from` / `value` turns with `<image>` placeholders paired with
`images`:

```json
{
  "source_name": "sharegpt4v_cap_100k",
  "conversations": [
    {"from": "human", "value": "<image>Describe this image in detail."},
    {"from": "gpt", "value": "A cat on a sofa."}
  ],
  "images": ["<user_png_bytes>"]
}
```

(`llava_video` uses the same `from` / `value` layout with `<image>` or `<video>`
paired with `videos`; `voice_assistant` pairs `"from": "human"` turns with
`audios`; text-only `tulu-3-sft-mixture` takes
`"conversations": {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.)

### 2. Generation (T2I — `imagenet1k`)

Comma-separated class labels in `conversations` paired with the target image in
`images` (emits a `user` text prompt and an `assistant` image with
`_img_tag: "gen"`):

```json
{
  "source_name": "imagenet1k",
  "conversations": "Sydney Opera House, opera house",
  "images": ["<gen_png_bytes>"]
}
```

### 3. Interleaved source & target images (`seed_edit_p23_multi_turn`)

ShareGPT-style turns where `human` `<image>` marks a user source image
(`role="user"`, `_img_tag="edit"`) and `gpt` `<image>` marks an assistant target
image (`role="assistant"`, `_img_tag="gen"`):

```json
{
  "source_name": "seed_edit_p23_multi_turn",
  "conversations": [
    {"from": "human", "value": "<image>Make the sky sunset orange."},
    {"from": "gpt", "value": "<image>"}
  ],
  "images": ["<user_png_bytes>", "<gen_png_bytes>"]
}
```

`images[0]` pairs with the `human` source placeholder; `images[1]` pairs with
the `gpt` target placeholder.

## After transform: `conversation_list`

`seedomni_transform` constructs a per-sample list of
:class:`~veomni.models.seed_omni.utils.conversation.ConversationItem` and runs
`OmniProcessor.preprocess(conversation_list)` in the Dataset stage (before
dynamic batching and `SeedOmniCollator`):

```python
ConversationItem(
    type="text" | "image" | "video" | "audio",
    value=torch.Tensor,  # token_ids for text; normalized patches/pixels/waveforms for media
    role="user" | "assistant",  # no system row; fold that text into a turn
    is_dummy=False | True,  # per-sample FSDP anchor injected when a modality is absent
    meta={
        "_omni_token_len": int,  # token length contributed by this item (used by dyn_bsz)
        "labels": torch.Tensor,  # on text items (IGNORE_INDEX = -100 for masked spans)
        ...,
    },
)
```

Because `OmniProcessor` runs inside `seedomni_transform`, each sample already
carries `_omni_token_len`, `labels`, and any per-sample `is_dummy=True`
placeholders when it reaches `DynamicBatchingSizeDataset` / `TextBatchingStrategy`:

- `dyn_bsz_count_mode: total` (and `physical_token_cap` via `get_length_fn`) sums
  `meta["_omni_token_len"]` across **all** items in `conversation_list`,
  **including** `is_dummy=True` placeholders.
- `dyn_bsz_count_mode: effective` (`get_length_by_labels_fn`) skips
  `is_dummy=True` items and sums `(meta["labels"] != -100).sum()` on text items
  plus `meta["_omni_token_len"]` on `role == "assistant"` media items (`0` on
  `role == "user"` media items).

`SeedOmniCollator` then groups the selected per-sample dicts into a micro-batch
`{"conversation_list": [sample_0, sample_1, ...]}` for `OmniModel.forward`
(see [Architecture](../design/architecture.md#3-training-flow)).

## Multisource training

In-tree smoke configs live under `configs/seed_omni/fake_model/train/base.yaml`.
For full model families (such as Janus on the `omniv2` branch / follow-up model
PRs, where data sources are declared in `configs/seed_omni/Janus/data.yaml`),
launch with:

```bash
# In-tree stand-in config:
bash train.sh tasks/omni/train_omni.py configs/seed_omni/fake_model/train/base.yaml

# Janus family config (on omniv2 / follow-up model PRs):
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Janus/janus_1.3b/train/base.yaml
```

See [Training and Inference](training_and_inference.md) for the full convert → train → resume → infer pipeline.

## Custom datasets

Implement a preprocessor in `veomni/data/seed_omni/preprocess.py` returning
`(constructed, media_refs)`, where `constructed` is the internal tuple form and
`media_refs` keys the sample's media refs **by the item type they pair with**:

```python
(
    [
        ["user", ("text", "..."), ("image", None), ("text", "...")],
        ["assistant", ("image", None), ("text", "...")],
    ],
    {"image": image_refs},   # one ref per ("image", None) entry, in order
)
```

A text-only sample returns `{}` for `media_refs`. Each modality key (`"image"`,
`"video"`, `"audio"`) is decoded through `MEDIA_FETCHERS`
(`veomni/data/seed_omni/utils/media.py`), which is shared with inference request
building (`OmniProcessor.__call__` → `fetch_media` → `build_conversation`) so
training and inference produce the same payload types and `item.meta` fields.

Register with `@SEED_OMNI_PREPROCESSOR_REGISTRY.register("your_source")`, set
`source_name: your_source` in the dataset config, route items by `role`, and
verify that the length of each `media_refs[modality]` list matches the number of
`(modality, None)` placeholders in `constructed` (see
`voice_assistant_preprocess`).
