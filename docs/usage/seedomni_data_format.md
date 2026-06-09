# SeedOmni V2 Data Format

This guide describes the **on-disk conversation schema** used by SeedOmni V2
(`data_type: seedomni`) and the Janus demo dataset builder. The design goal is
a **flat chat JSON** — a small number of `user` / `assistant` messages, each
with an ordered `content` list — so preprocess stays simple and the same sample
can feed understanding (I2T), generation (T2I), or mixed (UG) training.

## Pipeline overview

```
parquet row (conversations + images)
        │
        ▼  conv_preprocess("veomni_omni_demo", …)
        │  veomni/data/multimodal/preprocess.py
        ▼
seedomni_transform.process_seedomni_example
        │  veomni/data/multimodal/seedomni_transform.py
        ▼
raw_batch["conversation_list"]  →  OmniModel modules (SigLIP / VQVAE / …)
```

Set `data.data_type: seedomni` and register the sample with
`source_name: veomni_omni_demo` (or add your own preprocessor — see
[Preprocessor Registry](../key_features/preprocessor_registry.md)).

## On-disk row schema (parquet / jsonl)

Each training sample is one row with three fields:

| Column | Type | Description |
|--------|------|-------------|
| `source_name` | `str` | Preprocessor id, e.g. `veomni_omni_demo` |
| `conversations` | JSON bytes / list | Chat messages (see below) |
| `images` | `list[bytes]` | PNG/JPEG bytes, consumed **in conversation order** for every `{"type": "image"}` placeholder |

Image bytes are **not** embedded inside `conversations`. Each
`{"type": "image"}` item is a placeholder; `seedomni_transform` pairs them
with `images[0]`, `images[1]`, … in order.

## Message format (`veomni_omni_demo`)

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

- **Only two content types**: `text` and `image`. There is no separate
  `vq_image` type.
- **Text** carries the string in `"value"`.
- **Image** is a placeholder only (no inline path or bytes in JSON).
- **Understanding vs generation** is determined by **`role`**, not by item type:
  - `role == "user"` + `type == "image"` → SigLIP input (understanding)
  - `role == "assistant"` + `type == "image"` → VQVAE target (generation)

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
:class:`~veomni.models.seed_omni.conversation.ConversationItem`:

```python
ConversationItem(
    type="text" | "image",
    value=str | torch.Tensor,  # (C, H, W) uint8 for images
    role="system" | "user" | "assistant",
    meta={},  # empty at the data boundary; modules fill input_ids / labels later
)
```

Loss labels are set in the Janus text encoder during chat-template expand
(``meta["loss_mask"]`` on rows that differ from ``int(role == "assistant")``,
e.g. assistant prefix ``0``, boi/eoi/eos ``1``). Module-specific keys live in
``meta`` and are written during forward.

Modules read this list directly — chat template, tokenize, normalize, and
patchify happen inside each SeedOmni module at forward time (see `design.md`
§ "数据路由").

## Build a demo parquet

```bash
python scripts/multimodal/convert_data/make_janus_omni_demo.py \
    --dataset_mode understanding \
    --gen_image janus_out/infer_gen/generated_image_0.png \
    --und_reply janus_out/infer_und/reply.txt \
    --gen_prompt "A close-up high-contrast photo of Sydney Opera House at night." \
    --out_dir outputs/janus_demo/data \
    --num_repeat 32
```

Dataset modes (`--dataset_mode`):

| Mode | Rows emitted | Typical loss routing |
|------|----------------|----------------------|
| `understanding` | I2T only | `tok_decode` > 0, `vae_decode` == 0 (dummy) |
| `t2i` | T2I only | `vae_decode` > 0 |
| `mixed` | interleave UG | both heads > 0 |
| `all` | all three kinds | both heads > 0 on mixed rows |

Output file: ``<out_dir>/janus_omni_demo_<mode>.parquet``.

Verify loss routing (5-step overfit + checker):

```bash
bash scripts/seed_omni/verify_janus_demo_loss.sh understanding
bash scripts/seed_omni/verify_janus_demo_loss.sh t2i
bash scripts/seed_omni/verify_janus_demo_loss.sh mixed
```

8-GPU smoke train (20 steps by default):

```bash
bash scripts/seed_omni/debug_omni.sh understanding
bash scripts/seed_omni/debug_omni.sh t2i
bash scripts/seed_omni/debug_omni.sh mixed
```

Legacy flags:

| Flag | Effect |
|------|--------|
| `--include_interleave` | Alias for `--dataset_mode all` |
| `--only_interleave` | Alias for `--dataset_mode mixed` |
| `--num_repeat N` | Repeat each base row `N` times |

Point training at the output:

```yaml
data:
  data_type: seedomni
  train_path: outputs/janus_demo/data/janus_omni_demo_understanding.parquet
```

## Custom datasets

Implement a preprocessor in `veomni/data/multimodal/preprocess.py` that
returns the internal tuple form:

```python
[
    ["user", ("text", "..."), ("image", None), ("text", "...")],
    ["assistant", ("image", None), ("text", "...")],
]
```

Register with `@PREPROCESSOR_REGISTRY.register("your_source")` and set
`source_name: your_source` in the dataset config. Keep the same rules: one
`type="image"`, route by `role`, images in a parallel list consumed in order.
