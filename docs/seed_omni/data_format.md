# SeedOmni Data Format

This guide describes the **on-disk conversation schema** used by SeedOmni
(`data_type: seedomni`) and the multisource preprocessors under
``veomni/data/seed_omni/preprocess.py``. The design goal is a **flat chat JSON**
— a small number of `user` / `assistant` messages, each with an ordered
`content` list — so preprocess stays simple and the same sample can feed
understanding (I2T), generation (T2I), or mixed (UG) training.

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

Audio is decoded **at the clip's own sampling rate**, which the resulting item
declares in `meta["sampling_rate"]`. Do not resample in your source: the
modules that read a clip disagree on what rate they want (a Qwen3-Omni audio
tower runs at 16 kHz, the codec that turns assistant speech into talker targets
at 24 kHz) and each converts for itself. The rate is also what places a clip on
a shared wall clock against video under TMRoPE, so a wrong one is a silent
alignment error rather than an audio-quality one. wav / flac / ogg / mp3 decode
directly; m4a / aac / wma fall back to `audioread` (ffmpeg or gstreamer,
whichever the machine has).

An encoder's window caps clip length — 30 s for the Qwen3-Omni tower — and a
longer clip is **refused** rather than truncated, because a cut encoder input
paired with a text target covering the whole recording does not fail, it only
caps quality. Nothing catches that per sample, so it stops the run at whatever
step the row appears: check clip durations before pointing a new corpus at a
long training job, and segment into window-sized turns.

A source that stores raw sample arrays instead of encoded files carries no
header to read a rate from, and must state one via
`data.mm_configs.audio_sampling_rate`. Note what that setting is *not*: it
declares the rate the arrays are already at, and nothing resamples to it. It is
ignored for encoded refs, which state their own rate.

## Message format (ShareGPT4V-style sources)

Each message:

```json
{
  "role": "user" | "assistant" | "system",
  "content": [
    {"type": "text", "value": "..."},
    {"type": "image"}
  ]
}
```

Rules:

- **Content types**: `text`, `image`, `video`, `audio`. There is no separate
  `vq_image` type.
- **Text** carries the string in `"value"`.
- **Media** is a placeholder only (no inline path or bytes in JSON).
- **Input vs target** is determined by **`role`**, not by item type:
  - `role == "user"` + `type == "image"` → SigLIP input (understanding)
  - `role == "assistant"` + `type == "image"` → VQVAE target (generation)
  - `role == "user"` + `type == "audio"` → audio-encoder input
  - `role == "assistant"` + `type == "audio"` → speech a talker is trained to
    say, i.e. a label. It is held out of the language backbone's packed
    sequence entirely.

This matches the flat HF-style layout (`messages` + `content` array) used in
many chat datasets; VeOmni uses `"value"` for text instead of a type-specific
`"text"` key, and keeps pixels in the parallel `images` column.

## Three UG patterns

### 1. Understanding (I2T)

User sends image + question; assistant replies with text.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "Describe this image in detail."},
        {"type": "image"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "value": "A cat on a sofa."}
      ]
    }
  ],
  "images": ["<user_png_bytes>"]
}
```

### 2. Generation (T2I)

User sends prompt; assistant replies with an image target.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "A close-up photo of Sydney Opera House at night."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "image"}
      ]
    }
  ],
  "images": ["<gen_png_bytes>"]
}
```

### 3. Interleave (UG) — understanding + generation in one sample

Use **two messages total** (not four turns). User turn may interleave
`text → image → text`; assistant turn may interleave `image → text`.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "Describe this image in detail."},
        {"type": "image"},
        {"type": "text", "value": "A close-up photo of Sydney Opera House at night."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "image"},
        {"type": "text", "value": "The image shows …"}
      ]
    }
  ],
  "images": ["<user_png_bytes>", "<gen_png_bytes>"]
}
```

`images[0]` pairs the user-side placeholder; `images[1]` pairs the
assistant-side placeholder.

## After transform: `conversation_list`

`seedomni_transform` emits a per-sample list of
:class:`~veomni.models.seed_omni.utils.conversation.ConversationItem`:

```python
ConversationItem(
    type="text" | "image",
    value=str | torch.Tensor,  # (C, H, W) uint8 for images
    role="user" | "assistant",  # no system row; fold that text into a turn
    meta={},  # empty at the data boundary; modules fill input_ids / labels later
)
```

Loss labels are set in the Janus text encoder during chat-template expand
(``meta["loss_mask"]`` on rows that differ from ``int(role == "assistant")``,
e.g. assistant prefix ``0``, boi/eoi/eos ``1``). Module-specific keys live in
``meta`` and are written during forward.

Modules read this list directly — chat template, tokenize, normalize, and
patchify happen inside each SeedOmni module at forward time (see
`seed_omni.md` § 3).

## Janus multisource training

Data sources are declared in ``configs/seed_omni/Janus/data.yaml`` (ImageNet1k
T2I + ShareGPT4V caption I2T). Launch with the bundled YAML:

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Janus/janus_1.3b/train/base.yaml
```

See [`docs/seed_omni/example_models/janus.md`](example_models/janus.md)
for the full convert → train → resume → infer pipeline.

## Custom datasets

Implement a preprocessor in `veomni/data/seed_omni/preprocess.py` returning
`(constructed, image_refs, video_refs, audio_refs)`, where `constructed` is the
internal tuple form:

```python
(
    [
        ["user", ("text", "..."), ("image", None), ("text", "...")],
        ["assistant", ("image", None), ("text", "...")],
    ],
    image_refs,   # one per ("image", None), in order
    [],           # video_refs
    [],           # audio_refs
)
```

A 3-tuple without `audio_refs` is still accepted and read as "this source has
no audio", so a preprocessor written before audio existed keeps working.

Register with `@SEED_OMNI_PREPROCESSOR_REGISTRY.register("your_source")` and set
`source_name: your_source` in the dataset config. Keep the same rules: route by
`role`, and put media in a parallel list consumed in order — pairing is purely
positional, so your preprocessor owns the count matching and should raise if it
does not (see `voice_assistant_preprocess`).
