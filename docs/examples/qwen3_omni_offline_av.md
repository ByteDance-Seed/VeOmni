# Qwen3-Omni training with offline-extracted audio-enabled video

The default Qwen3-Omni recipe (`docs/examples/qwen3_omni_moe.md`) feeds raw
video files to the framework; VeOmni then decodes frames and pulls the audio
track out of the container at training time.

This recipe shows the **offline-A/V** path:

- each audio-enabled video has already been decoded into a list of frame
  bytes plus its matching audio bytes by a preceding data pipeline;
- the video and audio are still a single A/V unit aligned in time;
- the Qwen3-Omni processor interleaves their tokens via the standard
  `use_audio_in_video=True` path — same final input as the raw-video recipe,
  just with the decode work moved offline.

Use this when frame sampling, shot detection, or audio extraction is done
ahead of training and you don't want to re-decode every epoch.

> Looking for **independent** video + audio (e.g. a silent video plus an
> unrelated voice query)? That's a different shape — see the *Standalone
> audio turns* section below; the same recipe also accepts standalone audio
> via `sample["audios"]`.

---

## 1. Data shape

Each sample is a dict whose `videos` entries are **paired-A/V dicts**:

```json
{
  "videos": [
    {
      "frames": ["<png-bytes-frame-0>", "<png-bytes-frame-1>", "..."],
      "audio":  "<wav-bytes>",
      "video_fps": 2.0,
      "audio_fps": 16000
    }
  ],
  "conversations": [
    {"from": "human", "value": "<video>\nWhat is happening in the clip?"},
    {"from": "gpt",   "value": "Someone is speaking near a car."}
  ]
}
```

Key points:

- `videos[i]` is a dict with at least `frames` (`List[bytes]` of PNG/JPEG-encoded
  frames) and `audio` (WAV-encoded bytes, or a 1-D `np.ndarray` of mono
  samples). `video_fps` falls back to `mm_configs.fps`.  For `audio_fps`:
  - WAV bytes → `audio_fps` is read from the WAV header automatically; you
    can still override by setting `audio_fps` explicitly.
  - 1-D ndarray → `audio_fps` is **required** (we raise rather than silently
    drop the audio downstream).
- The single `<video>` marker per A/V item binds the entire dict — there is
  no separate `<audio>` marker for the paired audio. The Qwen3-Omni processor
  sees a non-empty per-video `audio_length` and emits an **interleaved**
  `<vision_bos><audio_bos> … <|video_pad|> / <|audio_pad|> chunks …
  <audio_eos><vision_eos>` sequence (the standard omni path).
- The registered preprocessor is `qwen_omni_offline_av` — set
  `data.source_name: qwen_omni_offline_av` in the YAML config.

### Standalone audio turns

If a turn carries audio that is **not** paired with any video (a voice query,
a sound effect, narration over a silent clip, …), use the existing
`sample["audios"]` field and an `<audio>` marker:

```json
{
  "videos": [{"frames": [...], "audio": "<wav-bytes>"}],
  "audios": ["<voice_query.wav>"],
  "conversations": [
    {"from": "human", "value": "<video>\nDescribe this. Also listen: <audio>"},
    {"from": "gpt",   "value": "..."}
  ]
}
```

`<video>` consumes the paired-A/V dict; `<audio>` consumes the standalone
clip — independent token positions. Same applies to `<image>` /
`sample["images"]`.

---

## 2. Offline extraction utility

To convert an existing `.mp4` corpus into the paired-A/V format above, decode
frames at the target fps and pull the audio track once per clip. Example using
`torchcodec` + `soundfile`:

```python
from io import BytesIO

import PIL.Image
import soundfile as sf
import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder


def extract_av(video_path: str, target_fps: float = 2.0, audio_sr: int = 16000) -> dict:
    # frames
    vdec = VideoDecoder(video_path, device="cpu")
    meta = vdec.metadata
    src_fps = meta.average_fps
    n = max(1, int(meta.num_frames * target_fps / src_fps))
    idx = torch.linspace(0, meta.num_frames - 1, n).round().long().tolist()
    frames_tensor = vdec.get_frames_at(idx).data  # (T, C, H, W) uint8 RGB
    frames_bytes = []
    for f in frames_tensor:
        img = PIL.Image.fromarray(f.permute(1, 2, 0).numpy())
        buf = BytesIO()
        img.save(buf, format="PNG")
        frames_bytes.append(buf.getvalue())

    # audio
    adec = AudioDecoder(video_path, sample_rate=audio_sr)
    audio_tensor = adec.get_all_samples().data.mean(dim=0)  # (T,) mono
    audio_buf = BytesIO()
    sf.write(audio_buf, audio_tensor.numpy(), audio_sr, format="WAV")
    audio_bytes = audio_buf.getvalue()

    return {
        "frames": frames_bytes,
        "audio": audio_bytes,
        "video_fps": target_fps,
        "audio_fps": audio_sr,
    }


def build_sample(video_path: str, prompt: str, answer: str) -> dict:
    return {
        "videos": [extract_av(video_path, target_fps=2.0)],
        "conversations": [
            {"from": "human", "value": f"<video>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }
```

Persist samples in any format your dataset class consumes (Parquet, JSONL
with base64-encoded bytes, Energon shards, …). If you store frames or audio
as files on disk, read them into bytes at sample-construction time — the
loader only consumes the in-memory bytes/ndarray shapes.

---

## 3. Configuration

A ready-made config lives at
`configs/multimodal/qwen3_omni/qwen3_omni_offline_av.yaml`. The key
behavioral differences from the default `qwen3_omni.yaml` are:

```yaml
data:
  source_name: qwen_omni_offline_av    # uses the new preprocessor
  mm_configs:
    use_audio_in_video: True            # paired A/V → interleaved tokens
```

Token interleaving is decided **per video position** by the processor at
`veomni/models/transformers/qwen3_omni_moe/processing_qwen3_omni_moe.py:159`
(`use_audio_in_video = audio_length != 0`). For the offline-A/V recipe each
video dict carries its own audio, so `audio_length > 0` and the per-position
decision is "interleave" — matching the raw-video path.

The shipped data manifest at
`configs/multimodal/data/qwen3_omni_offline_av.yaml` is a placeholder — open
it and replace `/path/to/your_offline_av_dataset` with your real source path.
Mix multiple sources by adding more entries to `sources` / `names` and
rebalancing `weights`.

---

## 4. Launching training

```bash
bash train.sh tasks/train_vlm.py \
    configs/multimodal/qwen3_omni/qwen3_omni_offline_av.yaml \
    --model.model_path Qwen3-Omni-30B-A3B-Instruct
```

Point `model_path` at the stock HF checkpoint — no offline MoE merge required.
The runtime `CheckpointTensorConverter` registered on the Qwen3-Omni modeling
class folds per-expert HF safetensor keys into VeOmni's fused
`gate_up_proj` / `down_proj` layout at load time. See
`docs/transformers_v5/transformers_v5_moe_weight_loading.md` for the format
matrix and how to convert a VeOmni-format checkpoint back to per-expert HF
keys for inference engines.

---

## 5. Smoke-testing the data path

A self-contained smoke test ships at
`tests/data/multimodal/test_qwen3_omni_offline_av.py`. It verifies:

1. The `qwen_omni_offline_av` preprocessor splits `<video>` / `<audio>` /
   `<image>` markers in declaration order.
2. `_dict_to_video_audio` on a `{"frames": [...], "audio": <wav-bytes>}`
   dict produces a 4-D video tensor and a non-empty mono audio array.
3. *(Optional, gated on `QWEN3_OMNI_MODEL_PATH`)* The full
   `process_sample_qwen_omni` pipeline produces an `input_ids` whose
   `<|video_pad|>` and `<|audio_pad|>` token runs are **interleaved** (the
   standard Qwen3-Omni omni layout), and `video_grid_thw` matches the
   sampled frame count.

Run:

```bash
# Cheap unit checks (no model weights needed)
pytest tests/data/multimodal/test_qwen3_omni_offline_av.py -v -s

# Full processor path (needs the Qwen3-Omni processor on disk)
QWEN3_OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
    pytest tests/data/multimodal/test_qwen3_omni_offline_av.py -v -s
```

The end-to-end test does **not** need the 30B model weights — only the
processor / tokenizer / config files from the HF checkpoint.
