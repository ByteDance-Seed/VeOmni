# Qwen3-Omni training with pre-extracted frames + standalone audio

The default Qwen3-Omni recipe (`docs/examples/qwen3_omni_moe.md`) feeds raw video
files to the framework; VeOmni then averages frames and extracts the audio track
from the video container. This recipe shows the **decoupled** path:

- **video** is supplied as a list of pre-extracted frame bytes (PNG/JPEG-encoded);
- **audio** is supplied as a standalone file/bytes;
- the processor emits video and audio token runs **without** interleaving
  (equivalent to `use_audio_in_video=False`).

Use this when you have your own frame sampler / shot-boundary detector, or when
your audio comes from a different source than the video.

---

## 1. Data shape

Each sample is a dict with three top-level keys:

```json
{
  "videos": [["<png-bytes-frame-0>", "<png-bytes-frame-1>", ...]],
  "audios": ["<path-or-bytes>"],
  "conversations": [
    {"from": "human", "value": "<video>\n<audio>\nWhat is happening in the clip?"},
    {"from": "gpt",   "value": "Someone is speaking near a car."}
  ]
}
```

Key points:

- `videos[i]` is a `List[bytes]` of encoded image frames (PNG or JPEG). The
  list-of-bytes branch in `veomni/data/multimodal/video_utils.py` reads each
  bytes payload into a PIL image, stacks them into a `(T, C, H, W)` tensor, and
  **does not** try to extract an audio track. The per-video audio slot is
  always `None`, which downstream signals "no audio paired with this video".
- `audios[i]` is independent from videos — either a path, raw bytes, or a numpy
  array. It is loaded by `fetch_audios()` and resampled to `mm_configs.sample_rate`.
- `conversations[k].value` may embed `<video>` and `<audio>` markers in any
  order; the registered preprocessor (`qwen_omni_decoupled_av`) splits them into
  separate content items. The order in `value` is the order the content items
  appear in the prompt, and it determines which entries from `videos` / `audios`
  bind to each marker (left-to-right).

### Why `<video>` and `<audio>` must both appear

Each `<video>` marker consumes one entry from `videos` (and one *paired-audio*
slot, which is `None` for frame-list inputs). Each `<audio>` marker consumes
one entry from `audios`. If you want the audio to be a separate, non-interleaved
token run, you **must** include both `<video>` and `<audio>` markers — otherwise
the standalone audio is never bound to a position in the conversation and the
processor sees no audio.

---

## 2. Frame extraction utility

To convert an existing `.mp4` corpus into the frame-bytes format above, sample
frames offline with the codec of your choice. Example using `torchcodec` at
2 fps:

```python
import json
from pathlib import Path
from io import BytesIO

import PIL.Image
import torch
from torchcodec.decoders import VideoDecoder


def extract_frames(video_path: str, target_fps: float = 2.0) -> list[bytes]:
    dec = VideoDecoder(video_path, device="cpu")
    meta = dec.metadata
    src_fps = meta.average_fps
    n = max(1, int(meta.num_frames * target_fps / src_fps))
    idx = torch.linspace(0, meta.num_frames - 1, n).round().long().tolist()
    frames = dec.get_frames_at(idx).data  # (T, C, H, W) uint8 RGB
    out = []
    for f in frames:
        img = PIL.Image.fromarray(f.permute(1, 2, 0).numpy())
        buf = BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def build_sample(video_path: str, audio_path: str, prompt: str, answer: str) -> dict:
    return {
        "videos": [extract_frames(video_path, target_fps=2.0)],
        "audios": [audio_path],
        "conversations": [
            {"from": "human", "value": f"<video>\n<audio>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }
```

Persist samples in any format your dataset class consumes (Parquet, JSONL with
base64-encoded bytes, Energon shards, etc.). If you store frames as files on
disk instead, load them into bytes at sample-construction time — the
`List[bytes]` branch is the only frame-list path currently wired up.

---

## 3. Configuration

A ready-made config lives at
`configs/multimodal/qwen3_omni/qwen3_omni_decoupled_av.yaml`. The two key
behavioral differences from the default `qwen3_omni.yaml` are:

```yaml
data:
  source_name: qwen_omni_decoupled_av    # uses the new preprocessor
  mm_configs:
    use_audio_in_video: False            # see note below
```

> **Note on `use_audio_in_video`.** For frame-list video inputs this flag is a
> no-op — audio is *always* unattached, since the frame-list loader has no audio
> track to extract. The flag still affects the raw-video paths/bytes loader
> (controls whether PyAV decodes the container's audio track). Setting it to
> `False` keeps the loader symmetric with the decoupled intent if you mix
> data sources.

Token interleaving is decided **per video position** by the processor at
`veomni/models/transformers/qwen3_omni_moe/processing_qwen3_omni_moe.py:159`
(`use_audio_in_video = audio_length != 0`). For frame-list videos `audio_length`
is always `0`, so the per-position decision is "no interleaving" regardless of
config.

The shipped data manifest at
`configs/multimodal/data/qwen3_omni_decoupled_av.yaml` is a placeholder — open
it and replace the `/path/to/your_decoupled_av_dataset` entry with your real
source path. Mix multiple sources by adding more entries to `sources` /
`names` and rebalancing `weights` (see
`configs/multimodal/data/tulu_sharegpt4v_llavavideo_voiceassistant.yaml` for a
multi-source example).

---

## 4. Launching training

```bash
bash train.sh tasks/train_vlm.py \
    configs/multimodal/qwen3_omni/qwen3_omni_decoupled_av.yaml \
    --model.model_path Qwen3-Omni-30B-A3B-Instruct-merge
```

(Use the same MoE-merged checkpoint as the default recipe.)

---

## 5. Smoke-testing the data path

A self-contained smoke test ships at
`tests/data/multimodal/test_qwen3_omni_decoupled_av.py`. It verifies:

1. The `qwen_omni_decoupled_av` preprocessor splits markers correctly.
2. `fetch_videos` on a `List[bytes]` returns `(video_tensor, audio=None)`.
3. `fetch_audios` independently loads standalone audio.
4. *(Optional, gated on `QWEN3_OMNI_MODEL_PATH`)* The full
   `process_sample_qwen_omni` pipeline produces an `input_ids` where the
   `<|video_pad|>` run and the `<|audio_pad|>` run are **distinct contiguous
   spans** (no interleaving), and `video_grid_thw` matches the frame count.

Run:

```bash
# Cheap unit checks (no model weights needed)
pytest tests/data/multimodal/test_qwen3_omni_decoupled_av.py -v -s

# Full processor path (needs the Qwen3-Omni processor on disk)
QWEN3_OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
    pytest tests/data/multimodal/test_qwen3_omni_decoupled_av.py -v -s
```

The end-to-end test does **not** need the 30B model weights — only the
processor / tokenizer / config files from the HF checkpoint.
