"""End-to-end preprocessing pipeline for LTX-2 offline training.

Integrates scene splitting, video captioning, text embedding computation,
and video/audio latent encoding into a single script.

Subcommands:
    split-scenes   Split raw videos into scene clips using PySceneDetect
    caption        Auto-caption video clips using a multimodal model
    preprocess     Compute text embeddings + VAE latents from a dataset file
    save-parquet   Pack precomputed .pt files into parquet for offline training
    all            Run the full pipeline: split → caption → preprocess

Usage examples:
    # Full pipeline
    python preprocess_dataset.py all \\
        --video_dir /path/to/raw/videos \\
        --data_dir /path/to/output \\
        --gemma_model_path /path/to/gemma3 \\
        --checkpoint_path /path/to/ltx2.safetensors \\
        --resolution_buckets 768x768x49

    # Only preprocess (text embeddings + VAE latents)
    python preprocess_dataset.py preprocess \\
        --dataset_file /path/to/dataset.json \\
        --gemma_model_path /path/to/gemma3 \\
        --checkpoint_path /path/to/ltx2.safetensors \\
        --resolution_buckets 768x768x49

    # Preprocess with reference videos for IC-LoRA
    python preprocess_dataset.py preprocess \\
        --dataset_file /path/to/dataset.json \\
        --gemma_model_path /path/to/gemma3 \\
        --checkpoint_path /path/to/ltx2.safetensors \\
        --resolution_buckets 768x768x49 \\
        --reference_column reference_path

Output directory structure (after full pipeline):
    data_dir/
    ├── .precomputed/
    │   ├── latents/              # VAE-encoded video latents (.pt)
    │   ├── conditions/           # Gemma text features (.pt)
    │   ├── audio_latents/        # VAE-encoded audio latents (.pt, optional)
    │   └── reference_latents/    # Reference video latents (.pt, optional)
    ├── clips/                    # Scene-split video clips
    └── dataset.json              # Captions + video paths
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 128
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _atomic_save(data: Any, out: Path) -> None:
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(out)


def _load_dataset_file(dataset_file: Path) -> list[dict]:
    if dataset_file.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(dataset_file).to_dict("records")
    elif dataset_file.suffix == ".json":
        with open(dataset_file, encoding="utf-8") as f:
            return json.load(f)
    elif dataset_file.suffix == ".jsonl":
        with open(dataset_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError(f"Unsupported dataset format: {dataset_file.suffix}")


def _save_dataset_file(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        if not records:
            return
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    elif output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    elif output_path.suffix == ".jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def _get_media_files(input_dir: Path, extensions: list[str] | None = None) -> list[Path]:
    if extensions is None:
        extensions = VIDEO_EXTENSIONS
    ext_set = {e.lower().lstrip(".") for e in extensions}
    return sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lstrip(".").lower() in ext_set)


def _load_paths_from_dataset(dataset_file: Path, column: str) -> list[Path]:
    """Load paths from a column in a CSV/JSON/JSONL dataset file."""
    import pandas as pd

    data_root = dataset_file.parent
    if dataset_file.suffix == ".csv":
        df = pd.read_csv(dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")
        return [data_root / Path(line.strip()) for line in df[column].tolist()]
    elif dataset_file.suffix == ".json":
        with open(dataset_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        paths = []
        for entry in data:
            if column not in entry:
                raise ValueError(f"Key '{column}' not found in JSON entry")
            paths.append(data_root / Path(entry[column].strip()))
        return paths
    elif dataset_file.suffix == ".jsonl":
        paths = []
        with open(dataset_file, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if column not in entry:
                    raise ValueError(f"Key '{column}' not found in JSONL entry")
                paths.append(data_root / Path(entry[column].strip()))
        return paths
    raise ValueError(f"Unsupported dataset format: {dataset_file.suffix}")


def _build_sharded_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    is_done: Callable[[int], bool],
    overwrite: bool,
) -> DataLoader | None:
    """Return a DataLoader over this rank's interleaved shard of *dataset*.

    Uses ``accelerate.PartialState`` for multi-GPU sharding. Items whose
    outputs already exist (per *is_done*) are filtered out unless *overwrite*.
    Returns ``None`` if this rank has nothing to do.
    """
    try:
        from accelerate import PartialState

        state = PartialState()
        rank, world = state.process_index, state.num_processes
    except ImportError:
        rank, world = 0, 1

    todo = [i for i in range(rank, len(dataset), world) if overwrite or not is_done(i)]
    if not todo:
        print(f"Rank {rank}/{world}: nothing to do")
        return None
    print(f"Rank {rank}/{world}: processing {len(todo):,} of {len(dataset):,} items")
    return DataLoader(Subset(dataset, todo), batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Stage 1: Scene splitting (requires: scenedetect, ffmpeg)
# ---------------------------------------------------------------------------


def split_scenes(
    video_dir: str,
    output_dir: str,
    detector: str = "content",
    threshold: float | None = None,
    min_scene_len: int | None = None,
    max_scenes: int | None = None,
    filter_shorter_than: str | None = None,
    duration: str | None = None,
    save_images: int = 0,
) -> list[Path]:
    """Split all videos in *video_dir* into scene clips saved under *output_dir*.

    Requires ``scenedetect`` and ``ffmpeg`` to be installed.
    Returns a list of paths to the generated clip files.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
        from scenedetect.scene_manager import save_images as save_scene_images
        from scenedetect.video_splitter import split_video_ffmpeg
    except ImportError:
        print("ERROR: scenedetect is required for scene splitting.")
        print("  pip install scenedetect[opencv]")
        sys.exit(1)

    from scenedetect.frame_timecode import FrameTimecode

    video_dir_path = Path(video_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = _get_media_files(video_dir_path)
    if not video_files:
        print(f"No video files found in {video_dir}")
        return []

    all_clips: list[Path] = []

    for video_file in tqdm(video_files, desc="Splitting scenes"):
        clip_dir = output_path / video_file.stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        video = open_video(str(video_file), backend="opencv")

        duration_tc = None
        if duration is not None:
            if duration.endswith("s"):
                duration_tc = FrameTimecode(timecode=float(duration[:-1]), fps=video.frame_rate)
            elif ":" in duration:
                duration_tc = FrameTimecode(timecode=duration, fps=video.frame_rate)
            else:
                duration_tc = FrameTimecode(timecode=int(duration), fps=video.frame_rate)

        filter_tc = None
        if filter_shorter_than is not None:
            if filter_shorter_than.endswith("s"):
                filter_tc = FrameTimecode(timecode=float(filter_shorter_than[:-1]), fps=video.frame_rate)
            elif ":" in filter_shorter_than:
                filter_tc = FrameTimecode(timecode=filter_shorter_than, fps=video.frame_rate)
            else:
                filter_tc = FrameTimecode(timecode=int(filter_shorter_than), fps=video.frame_rate)

        scene_manager = SceneManager()
        kwargs: dict[str, Any] = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if min_scene_len is not None:
            kwargs["min_scene_len"] = min_scene_len

        if detector == "content":
            scene_manager.add_detector(ContentDetector(**kwargs))
        else:
            from scenedetect import AdaptiveDetector, ThresholdDetector

            if detector == "adaptive":
                if "threshold" in kwargs:
                    kwargs["adaptive_threshold"] = kwargs.pop("threshold")
                scene_manager.add_detector(AdaptiveDetector(**kwargs))
            elif detector == "threshold":
                scene_manager.add_detector(ThresholdDetector(**kwargs))
            else:
                scene_manager.add_detector(ContentDetector(**kwargs))

        scene_manager.detect_scenes(video=video, show_progress=True, duration=duration_tc)
        scenes = scene_manager.get_scene_list()

        if filter_tc:
            scenes = [(s, e) for s, e in scenes if (e.get_frames() - s.get_frames()) >= filter_tc.get_frames()]

        if max_scenes and len(scenes) > max_scenes:
            scenes = scenes[:max_scenes]

        print(f"  {video_file.name}: {len(scenes)} scenes detected")

        split_video_ffmpeg(
            input_video_path=str(video_file),
            scene_list=scenes,
            output_dir=str(clip_dir),
            show_progress=True,
        )

        if save_images > 0:
            save_scene_images(
                scene_list=scenes,
                video=video,
                num_images=save_images,
                output_dir=str(clip_dir),
                show_progress=True,
            )

        clips = sorted(clip_dir.glob("*.mp4"))
        all_clips.extend(clips)

    print(f"Total clips generated: {len(all_clips)}")
    return all_clips


# ---------------------------------------------------------------------------
# Stage 2: Video captioning (requires: transformers or external captioner)
# ---------------------------------------------------------------------------


def caption_videos(
    input_dir: str,
    output: str,
    captioner_type: str = "qwen_omni",
    device: str | None = None,
    instruction: str | None = None,
    fps: int = 3,
    include_audio: bool = True,
    extensions: list[str] | None = None,
) -> Path:
    """Generate captions for all videos in *input_dir*.

    Supports two captioner backends:
    - ``qwen_omni``: local Qwen2.5-Omni model (requires ``ltx_trainer``)
    - ``gemini_flash``: Google Gemini API (requires ``GOOGLE_API_KEY``)

    Returns the path to the generated dataset file.
    """
    input_path = Path(input_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    media_files = _get_media_files(input_path, extensions)
    if not media_files:
        print(f"No media files found in {input_dir}")
        return output_path

    print(f"Found {len(media_files)} media files to caption")

    if captioner_type == "gemini_flash":
        _caption_with_gemini(media_files, output_path, instruction, fps, include_audio)
    else:
        _caption_with_ltx_trainer(
            media_files,
            output_path,
            captioner_type,
            device,
            instruction,
            fps,
            include_audio,
        )

    return output_path


def _caption_with_gemini(
    media_files: list[Path],
    output_path: Path,
    instruction: str | None,
    fps: int,
    include_audio: bool,
) -> None:
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: google-generativeai is required for Gemini captioning.")
        print("  pip install google-generativeai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    default_instruction = (
        "Describe this video in detail, including the visual content, actions, "
        "camera movements, and any audio or dialogue."
    )
    prompt_text = instruction or default_instruction

    records: list[dict] = []
    for media_file in tqdm(media_files, desc="Captioning"):
        try:
            import base64

            with open(media_file, "rb") as f:
                video_bytes = f.read()

            mime_type = f"video/{media_file.suffix.lstrip('.')}"
            if media_file.suffix.lower() in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif media_file.suffix.lower() == ".png":
                mime_type = "image/png"

            response = model.generate_content(
                [
                    prompt_text,
                    {"mime_type": mime_type, "data": base64.b64encode(video_bytes).decode()},
                ]
            )
            caption = response.text.strip()
        except Exception as e:
            print(f"  WARNING: Failed to caption {media_file.name}: {e}")
            caption = ""

        records.append({"caption": caption, "media_path": str(media_file.name)})

    _save_dataset_file(records, output_path)
    print(f"Captions saved to {output_path}")


def _caption_with_ltx_trainer(
    media_files: list[Path],
    output_path: Path,
    captioner_type: str,
    device: str | None,
    instruction: str | None,
    fps: int,
    include_audio: bool,
) -> None:
    try:
        from ltx_trainer.captioning import CaptionerType, create_captioner
    except ImportError:
        print("ERROR: ltx_trainer is required for local captioning.")
        print("  pip install ltx-trainer")
        sys.exit(1)

    device_str = device or ("cuda" if IS_CUDA_AVAILABLE else "cpu")
    ct = CaptionerType(captioner_type)
    captioner = create_captioner(captioner_type=ct, device=device_str, instruction=instruction)

    records: list[dict] = []
    for media_file in tqdm(media_files, desc="Captioning"):
        try:
            caption = captioner.caption(
                path=media_file,
                fps=fps,
                include_audio=include_audio,
                clean_caption=True,
            )
        except Exception as e:
            print(f"  WARNING: Failed to caption {media_file.name}: {e}")
            caption = ""

        records.append({"caption": caption, "media_path": str(media_file.name)})

    _save_dataset_file(records, output_path)
    print(f"Captions saved to {output_path}")


# ---------------------------------------------------------------------------
# Stage 3a: Text embedding computation (Gemma + FeatureExtractor)
# ---------------------------------------------------------------------------


COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    "This video shows",
    "The video shows",
    "This clip shows",
    "The clip shows",
    "This video depicts",
    "The video depicts",
    "This video captures",
    "The video captures",
    "This video features",
    "The video features",
    "This image shows",
    "The image shows",
)


def _clean_llm_prefix(text: str) -> str:
    text = text.strip()
    for phrase in COMMON_LLM_START_PHRASES:
        if text.startswith(phrase):
            text = text.removeprefix(phrase).strip()
            break
    return text


class CaptionsDataset(Dataset):
    """Dataset for processing text captions.

    Loads captions from CSV/JSON/JSONL, computes output embedding paths,
    and optionally applies LoRA trigger words and LLM prefix cleaning.
    """

    def __init__(
        self,
        dataset_file: str | Path,
        caption_column: str,
        media_column: str = "media_path",
        lora_trigger: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.caption_column = caption_column
        self.media_column = media_column
        self.lora_trigger = f"{lora_trigger.strip()} " if lora_trigger else ""

        self.caption_data = self._load_caption_data()
        self.output_paths = list(self.caption_data.keys())
        self.prompts = list(self.caption_data.values())

        if remove_llm_prefixes:
            self._clean_llm_prefixes()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prompt = self.lora_trigger + self.prompts[index]
        return {
            "prompt": prompt,
            "output_path": self.output_paths[index],
            "index": index,
        }

    def _load_caption_data(self) -> dict[str, str]:
        if self.dataset_file.suffix == ".csv":
            return self._load_from_csv()
        elif self.dataset_file.suffix == ".json":
            return self._load_from_json()
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_from_jsonl()
        raise ValueError(f"Unsupported dataset format: {self.dataset_file.suffix}")

    def _load_from_csv(self) -> dict[str, str]:
        import pandas as pd

        df = pd.read_csv(self.dataset_file)
        if self.caption_column not in df.columns:
            raise ValueError(f"Column '{self.caption_column}' not found in CSV")
        if self.media_column not in df.columns:
            raise ValueError(f"Column '{self.media_column}' not found in CSV")
        caption_data = {}
        for _, row in df.iterrows():
            media_path = Path(row[self.media_column].strip())
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = row[self.caption_column]
        return caption_data

    def _load_from_json(self) -> dict[str, str]:
        with open(self.dataset_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        caption_data = {}
        for entry in data:
            if self.caption_column not in entry:
                raise ValueError(f"Key '{self.caption_column}' not found in JSON entry")
            if self.media_column not in entry:
                raise ValueError(f"Key '{self.media_column}' not found in JSON entry")
            media_path = Path(entry[self.media_column].strip())
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = entry[self.caption_column]
        return caption_data

    def _load_from_jsonl(self) -> dict[str, str]:
        caption_data = {}
        with open(self.dataset_file, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if self.caption_column not in entry:
                    raise ValueError(f"Key '{self.caption_column}' not found in JSONL entry")
                if self.media_column not in entry:
                    raise ValueError(f"Key '{self.media_column}' not found in JSONL entry")
                media_path = Path(entry[self.media_column].strip())
                output_path = str(media_path.with_suffix(".pt"))
                caption_data[output_path] = entry[self.caption_column]
        return caption_data

    def _clean_llm_prefixes(self) -> None:
        for i in range(len(self.prompts)):
            self.prompts[i] = self.prompts[i].strip()
            for phrase in COMMON_LLM_START_PHRASES:
                if self.prompts[i].startswith(phrase):
                    self.prompts[i] = self.prompts[i].removeprefix(phrase).strip()
                    break


def load_feature_extractor(checkpoint_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Load feature extractor weights from LTX-2 checkpoint."""
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        EMBEDDINGS_PROCESSOR_KEY_REMAP,
        _create_feature_extractor,
    )
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_path)

    transformer_config_path = (
        ckpt_path / "transformer" / "config.json"
        if ckpt_path.is_dir()
        else ckpt_path.parent / "transformer" / "config.json"
    )
    if not transformer_config_path.exists():
        transformer_config_path = ckpt_path / "config.json" if ckpt_path.is_dir() else ckpt_path.parent / "config.json"

    with open(transformer_config_path) as f:
        config = json.load(f)

    transformer_config = config.get("transformer", config)
    feature_extractor = _create_feature_extractor(transformer_config)

    fe_prefixes = {
        old_prefix: new_prefix
        for old_prefix, new_prefix in EMBEDDINGS_PROCESSOR_KEY_REMAP.items()
        if new_prefix.startswith("feature_extractor.")
    }
    target_prefixes = tuple(fe_prefixes.keys())

    fe_state_dict = {}

    if ckpt_path.is_dir():
        safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
    elif ckpt_path.suffix == ".safetensors":
        safetensor_files = [ckpt_path]
    else:
        safetensor_files = []

    if safetensor_files:
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt", device=str(device)) as f:
                for key in f.keys():
                    if key.startswith(target_prefixes):
                        for old_prefix in fe_prefixes:
                            if key.startswith(old_prefix):
                                fe_key = "feature_extractor." + key[len(old_prefix) :]
                                fe_state_dict[fe_key] = f.get_tensor(key).to(dtype)
                                break
    else:
        if ckpt_path.is_dir():
            bin_files = sorted(ckpt_path.glob("*.bin")) + sorted(ckpt_path.glob("*.pt"))
        elif ckpt_path.suffix in (".bin", ".pt"):
            bin_files = [ckpt_path]
        else:
            bin_files = []

        for bin_path in bin_files:
            all_state_dict = torch.load(str(bin_path), map_location=device, weights_only=True)
            for key, value in all_state_dict.items():
                for old_prefix in fe_prefixes:
                    if key.startswith(old_prefix):
                        fe_key = "feature_extractor." + key[len(old_prefix) :]
                        fe_state_dict[fe_key] = value.to(dtype)
                        break
            del all_state_dict

    if fe_state_dict:
        missing, unexpected = feature_extractor.load_state_dict(fe_state_dict, strict=False)
        if missing:
            print(f"Feature extractor missing keys: {missing}")
        if unexpected:
            print(f"Feature extractor unexpected keys: {unexpected}")
    else:
        print("WARNING: No feature extractor weights found in checkpoint!")

    return feature_extractor.to(device=device, dtype=dtype)


def compute_caption_embeddings(
    dataset_file: str,
    output_dir: str,
    checkpoint_path: str,
    gemma_model_path: str,
    caption_column: str = "caption",
    media_column: str = "media_path",
    max_sequence_length: int = 256,
    lora_trigger: str | None = None,
    remove_llm_prefixes: bool = False,
    batch_size: int = 1,
    device: str | None = None,
    load_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    """Encode captions through Gemma + FeatureExtractor and save to disk.

    Uses ``CaptionsDataset`` + ``DataLoader`` with multi-GPU sharding via
    ``accelerate.PartialState``. Already-computed outputs are skipped unless
    *overwrite* is True; writes are atomic so interrupted runs are safe to resume.
    """
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

    device_str = device or get_device_type()
    dev = torch.device(device_str)
    dtype = torch.bfloat16

    dataset = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
    )
    print(f"Loaded {len(dataset):,} captions")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if batch_size > 1:
        print("WARNING: Gemma tokenizer does not support batching. Overriding batch_size to 1.")
        batch_size = 1

    dataloader = _build_sharded_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        is_done=lambda idx: (output_path / dataset.output_paths[idx]).is_file(),
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    print("Loading Gemma text encoder...")
    text_encoder = GemmaTextEncoder.from_pretrained(gemma_model_path, max_length=max_sequence_length)
    if load_in_8bit:
        text_encoder = text_encoder.to(device=dev)
    else:
        text_encoder = text_encoder.to(device=dev, dtype=dtype)
    text_encoder.eval()

    print("Loading feature extractor from checkpoint...")
    feature_extractor = load_feature_extractor(checkpoint_path, dev, dtype)
    feature_extractor.eval()

    print(f"Processing captions in {len(dataloader):,} batches...")

    for batch in tqdm(dataloader, desc="Encoding captions"):
        with torch.no_grad():
            for i in range(len(batch["prompt"])):
                hidden_states, prompt_attention_mask = text_encoder.encode(batch["prompt"][i], padding_side="left")
                video_feats, audio_feats = feature_extractor(hidden_states, prompt_attention_mask, padding_side="left")

                output_rel_path = Path(batch["output_path"][i])
                out_dir = output_path / output_rel_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)

                embedding_data: dict[str, torch.Tensor] = {
                    "video_prompt_embeds": video_feats[0].cpu().contiguous(),
                    "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
                }
                if audio_feats is not None:
                    embedding_data["audio_prompt_embeds"] = audio_feats[0].cpu().contiguous()

                output_file = output_path / output_rel_path
                _atomic_save(embedding_data, output_file)

    print(f"Caption embeddings saved to {output_path}")


# ---------------------------------------------------------------------------
# Stage 3b: Video latent computation (VAE encoder)
# ---------------------------------------------------------------------------


def parse_resolution_buckets(buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse ``"WxHxF;WxHxF;..."`` into a list of ``(frames, height, width)`` tuples."""
    buckets = []
    for bucket_str in buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))
        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}")
        if f % VAE_TEMPORAL_FACTOR != 1:
            raise ValueError(f"Frames must be 8k+1, got {f}")
        buckets.append((f, h, w))
    return buckets


def compute_scaled_resolution_buckets(
    resolution_buckets: list[tuple[int, int, int]],
    scale_factor: int,
) -> list[tuple[int, int, int]]:
    """Compute scaled resolution buckets for IC-LoRA reference videos."""
    if scale_factor == 1:
        return resolution_buckets

    scaled_buckets = []
    for frames, height, width in resolution_buckets:
        if height % scale_factor != 0:
            raise ValueError(f"Height {height} not evenly divisible by scale factor {scale_factor}")
        if width % scale_factor != 0:
            raise ValueError(f"Width {width} not evenly divisible by scale factor {scale_factor}")

        scaled_h = height // scale_factor
        scaled_w = width // scale_factor

        if scaled_h % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(f"Scaled height {scaled_h} not divisible by {VAE_SPATIAL_FACTOR}")
        if scaled_w % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(f"Scaled width {scaled_w} not divisible by {VAE_SPATIAL_FACTOR}")

        scaled_buckets.append((frames, scaled_h, scaled_w))

    return scaled_buckets


def _read_video_frames(video_path: Path, max_frames: int) -> tuple[torch.Tensor, float]:
    """Read video frames as ``[F, C, H, W]`` float tensor in ``[0, 1]``."""
    try:
        from torchvision.io import read_video as tv_read_video

        video, _audio, info = tv_read_video(str(video_path), pts_unit="sec")
        fps = info.get("video_fps", 24.0)
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)
        video = video[:max_frames]
        return video, fps
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        frames = iio.imread(str(video_path), plugin="pyav")
        fps = 24.0
        try:
            props = iio.props(str(video_path), plugin="pyav")
            fps = props.get("fps", 24.0)
        except Exception:
            pass
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        if frames_tensor.dim() == 3:
            frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        frames_tensor = frames_tensor[:max_frames]
        return frames_tensor, fps
    except ImportError:
        pass

    raise RuntimeError(f"Cannot read video {video_path}. Install torchvision or imageio[pyav].")


def _get_video_frame_count(video_path: Path) -> int:
    """Get frame count without loading all frames."""
    try:
        from torchvision.io import read_video as tv_read_video

        _v, _a, info = tv_read_video(str(video_path), pts_unit="sec")
        return info.get("video_frames", 0)
    except Exception:
        pass
    try:
        import imageio.v3 as iio

        props = iio.props(str(video_path), plugin="pyav")
        return props.get("n_frames", 0)
    except Exception:
        pass
    return 0


def _resize_and_crop(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
    reshape_mode: str = "center",
) -> torch.Tensor:
    """Resize video ``[F, C, H, W]`` to target dimensions."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import crop, resize

    _f, _c, cur_h, cur_w = video.shape
    cur_aspect = cur_w / cur_h
    target_aspect = target_width / target_height

    if cur_aspect > target_aspect:
        new_w = int(cur_w * target_height / cur_h)
        video = resize(video, [target_height, new_w], interpolation=InterpolationMode.BICUBIC)
    else:
        new_h = int(cur_h * target_width / cur_w)
        video = resize(video, [new_h, target_width], interpolation=InterpolationMode.BICUBIC)

    _f, _c, cur_h, cur_w = video.shape
    delta_h = cur_h - target_height
    delta_w = cur_w - target_width

    if reshape_mode == "random":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    else:
        top, left = delta_h // 2, delta_w // 2

    video = crop(video, top=top, left=left, height=target_height, width=target_width)
    return video


def _select_bucket(
    num_frames: int,
    height: int,
    width: int,
    buckets: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    relevant = [b for b in buckets if b[0] <= num_frames]
    if not relevant:
        raise ValueError(f"No bucket has <= {num_frames} frames. Buckets: {buckets}")

    def distance(bucket: tuple[int, int, int]) -> tuple:
        bf, bh, bw = bucket
        return (
            abs(math.log(width / height) - math.log(bw / bh)),
            -bf,
            -(bh * bw),
        )

    return min(relevant, key=distance)


class MediaDataset(Dataset):
    """Dataset for processing video files with resolution bucket selection.

    Loads videos from CSV/JSON/JSONL metadata, applies resize/crop transforms,
    handles resolution bucket matching, and optionally extracts audio.
    """

    def __init__(
        self,
        dataset_file: str | Path,
        main_media_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
        with_audio: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode
        self.with_audio = with_audio

        self.main_media_paths = _load_paths_from_dataset(self.dataset_file, main_media_column)
        self.video_paths = _load_paths_from_dataset(self.dataset_file, video_column)

        self._filter_valid_videos()

        self.max_target_frames = max(self.resolution_buckets, key=lambda x: x[0])[0]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, list):
            return index

        video_path: Path = self.video_paths[index]
        data_root = self.dataset_file.parent
        relative_path = str(video_path.relative_to(data_root))
        media_relative_path = str(self.main_media_paths[index].relative_to(data_root))

        video_frames, fps = _read_video_frames(video_path, self.max_target_frames)
        num_frames, _c, h, w = video_frames.shape

        target_f, target_h, target_w = _select_bucket(num_frames, h, w, self.resolution_buckets)
        video_frames = _resize_and_crop(video_frames[:target_f], target_h, target_w, self.reshape_mode)

        video_frames = video_frames.clamp(0, 1)
        video_frames = (video_frames - 0.5) / 0.5

        video_cfhw = video_frames.permute(1, 0, 2, 3).contiguous()
        _, num_frames_out, height_out, width_out = video_cfhw.shape

        result: dict[str, Any] = {
            "video": video_cfhw,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": num_frames_out,
                "height": height_out,
                "width": width_out,
                "fps": fps,
            },
        }

        if self.with_audio:
            target_duration = num_frames_out / fps
            audio_data = self._extract_audio(video_path, target_duration)
            if audio_data is not None:
                result["audio"] = audio_data

        return result

    @staticmethod
    def _extract_audio(video_path: Path, target_duration: float) -> dict[str, Any] | None:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(video_path))
            target_samples = int(target_duration * sample_rate)
            current_samples = waveform.shape[-1]

            if current_samples > target_samples:
                waveform = waveform[..., :target_samples]
            elif current_samples < target_samples:
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception:
            return None

    def _filter_valid_videos(self) -> None:
        original_length = len(self.video_paths)
        valid_video_paths = []
        valid_main_media_paths = []
        min_frames_required = min(self.resolution_buckets, key=lambda x: x[0])[0]

        for i, video_path in enumerate(self.video_paths):
            if not video_path.is_file():
                continue
            try:
                frame_count = _get_video_frame_count(video_path)
                if frame_count >= min_frames_required or frame_count == 0:
                    valid_video_paths.append(video_path)
                    valid_main_media_paths.append(self.main_media_paths[i])
                else:
                    print(f"  Skipping {video_path} — {frame_count} frames < {min_frames_required}")
            except Exception:
                valid_video_paths.append(video_path)
                valid_main_media_paths.append(self.main_media_paths[i])

        self.video_paths = valid_video_paths
        self.main_media_paths = valid_main_media_paths

        if len(self.video_paths) < original_length:
            print(
                f"  Filtered out {original_length - len(self.video_paths)} videos. "
                f"Proceeding with {len(self.video_paths)} valid videos."
            )


def encode_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
    use_tiling: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> dict[str, Any]:
    """Encode video into latent representation.

    Args:
        vae: Video VAE encoder model
        video: ``[B, C, F, H, W]`` tensor
        use_tiling: Whether to use spatial tiling for memory efficiency
        tile_size: Tile size in pixels (must be divisible by 32)
        tile_overlap: Overlap between tiles in pixels
    """
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    if video.ndim == 4:
        video = video.unsqueeze(0)

    video = video.to(device=device, dtype=vae_dtype)

    if use_tiling:
        latents = tiled_encode_video(vae, video, tile_size, tile_overlap)
    else:
        latents = vae(video)

    _, _, num_frames, height, width = latents.shape

    return {
        "latents": latents,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def tiled_encode_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> torch.Tensor:
    """Encode video using spatial tiling for memory efficiency.

    Splits the video into overlapping spatial tiles, encodes each tile
    separately, and blends the results using linear feathering in the
    overlap regions.
    """
    batch, _channels, frames, height, width = video.shape
    device = video.device
    dtype = video.dtype

    if tile_size % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_size must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_size}")
    if tile_overlap % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_overlap must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_overlap}")
    if tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")

    if height <= tile_size and width <= tile_size:
        return vae(video)

    output_height = height // VAE_SPATIAL_FACTOR
    output_width = width // VAE_SPATIAL_FACTOR
    output_frames = 1 + (frames - 1) // VAE_TEMPORAL_FACTOR
    latent_channels = 128

    output = torch.zeros(
        (batch, latent_channels, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )
    weights = torch.zeros(
        (batch, 1, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )

    step_h = tile_size - tile_overlap
    step_w = tile_size - tile_overlap

    h_positions = list(range(0, max(1, height - tile_overlap), step_h))
    w_positions = list(range(0, max(1, width - tile_overlap), step_w))

    if h_positions[-1] + tile_size < height:
        h_positions.append(height - tile_size)
    if w_positions[-1] + tile_size < width:
        w_positions.append(width - tile_size)

    h_positions = sorted(set(h_positions))
    w_positions = sorted(set(w_positions))

    overlap_out_h = tile_overlap // VAE_SPATIAL_FACTOR
    overlap_out_w = tile_overlap // VAE_SPATIAL_FACTOR

    for h_pos in h_positions:
        for w_pos in w_positions:
            h_start = max(0, h_pos)
            w_start = max(0, w_pos)
            h_end = min(h_start + tile_size, height)
            w_end = min(w_start + tile_size, width)

            tile_h = ((h_end - h_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR
            tile_w = ((w_end - w_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR

            if tile_h < VAE_SPATIAL_FACTOR or tile_w < VAE_SPATIAL_FACTOR:
                continue

            h_end = h_start + tile_h
            w_end = w_start + tile_w

            tile = video[:, :, :, h_start:h_end, w_start:w_end]
            encoded_tile = vae(tile)

            _, _, tile_out_frames, tile_out_height, tile_out_width = encoded_tile.shape

            out_h_start = h_start // VAE_SPATIAL_FACTOR
            out_w_start = w_start // VAE_SPATIAL_FACTOR
            out_h_end = min(out_h_start + tile_out_height, output_height)
            out_w_end = min(out_w_start + tile_out_width, output_width)

            actual_tile_h = out_h_end - out_h_start
            actual_tile_w = out_w_end - out_w_start
            encoded_tile = encoded_tile[:, :, :, :actual_tile_h, :actual_tile_w]

            mask = torch.ones(
                (1, 1, tile_out_frames, actual_tile_h, actual_tile_w),
                device=device,
                dtype=dtype,
            )

            if h_pos > 0 and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :overlap_out_h, :] *= fade_in.view(1, 1, 1, -1, 1)

            if h_end < height and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, -overlap_out_h:, :] *= fade_out.view(1, 1, 1, -1, 1)

            if w_pos > 0 and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, :overlap_out_w] *= fade_in.view(1, 1, 1, 1, -1)

            if w_end < width and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, -overlap_out_w:] *= fade_out.view(1, 1, 1, 1, -1)

            output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += encoded_tile * mask
            weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += mask

    output = output / (weights + 1e-8)
    return output


def encode_audio(
    audio_vae_encoder: torch.nn.Module,
    audio_processor: Any,
    waveform: torch.Tensor,
    sampling_rate: int,
) -> dict[str, Any]:
    """Encode audio waveform into latent representation.

    Args:
        audio_vae_encoder: Audio VAE encoder model
        audio_processor: AudioProcessor for waveform-to-spectrogram conversion
        waveform: ``[channels, samples]`` tensor
        sampling_rate: Audio sampling rate
    """
    from ltx_core.types import Audio

    device = next(audio_vae_encoder.parameters()).device
    dtype = next(audio_vae_encoder.parameters()).dtype

    waveform = waveform.to(device=device, dtype=dtype)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    duration = waveform.shape[-1] / sampling_rate

    mel = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sampling_rate))
    mel = mel.to(dtype=dtype)

    latents = audio_vae_encoder(mel)
    _, _channels, time_steps, freq_bins = latents.shape

    return {
        "latents": latents.squeeze(0).cpu().contiguous(),
        "num_time_steps": time_steps,
        "frequency_bins": freq_bins,
        "duration": duration,
    }


def compute_video_latents(
    dataset_file: str,
    output_dir: str,
    checkpoint_path: str,
    resolution_buckets: list[tuple[int, int, int]],
    video_column: str = "media_path",
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str | None = None,
    vae_tiling: bool = False,
    with_audio: bool = False,
    audio_output_dir: str | None = None,
    overwrite: bool = False,
) -> None:
    """Encode videos through VAE and save latent representations.

    Uses ``MediaDataset`` + ``DataLoader`` with multi-GPU sharding via
    ``accelerate.PartialState``. Already-computed outputs are skipped unless
    *overwrite* is True; writes are atomic so interrupted runs are safe to resume.
    """
    from veomni.models.diffusers.ltx2_3.ltx_core.model.video_vae import load_video_encoder

    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")

    device_str = device or get_device_type()
    dev = torch.device(device_str)

    dataset = MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
    )
    print(f"Loaded {len(dataset)} valid media files")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audio_output_path: Path | None = None
    if with_audio:
        audio_output_path = Path(audio_output_dir)
        audio_output_path.mkdir(parents=True, exist_ok=True)

    if with_audio and batch_size > 1:
        print("WARNING: Audio processing requires batch_size=1. Overriding.")
        batch_size = 1

    data_root = dataset.dataset_file.parent

    def _is_done(idx: int) -> bool:
        rel = dataset.main_media_paths[idx].relative_to(data_root).with_suffix(".pt")
        if not (output_path / rel).is_file():
            return False
        return audio_output_path is None or (audio_output_path / rel).is_file()

    dataloader = _build_sharded_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        is_done=_is_done,
        overwrite=overwrite,
    )
    if dataloader is None:
        return

    print("Loading video VAE encoder...")
    vae = load_video_encoder(checkpoint_path, device=dev, dtype=torch.bfloat16)
    vae.eval()

    audio_vae_encoder = None
    audio_processor = None
    if with_audio:
        try:
            from ltx_core.model.audio_vae import AudioProcessor, load_audio_encoder
        except ImportError:
            from ltx_core.model.audio_vae import AudioProcessor
            from ltx_trainer.model_loader import load_audio_vae_encoder as load_audio_encoder

        print("Loading audio VAE encoder...")
        audio_vae_encoder = load_audio_encoder(checkpoint_path, device=dev, dtype=torch.float32)
        audio_vae_encoder.eval()

        audio_processor = AudioProcessor(
            target_sample_rate=audio_vae_encoder.sample_rate,
            mel_bins=audio_vae_encoder.mel_bins,
            mel_hop_length=audio_vae_encoder.mel_hop_length,
            n_fft=audio_vae_encoder.n_fft,
        ).to(dev)

    audio_success_count = 0
    audio_skip_count = 0

    for batch in tqdm(dataloader, desc="Encoding videos"):
        video = batch["video"]

        with torch.no_grad():
            video_latent_data = encode_video(vae=vae, video=video, use_tiling=vae_tiling)

        for i in range(len(batch["relative_path"])):
            output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
            output_file = output_path / output_rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            latent_data = {
                "latents": video_latent_data["latents"][i].cpu().contiguous(),
                "num_frames": video_latent_data["num_frames"],
                "height": video_latent_data["height"],
                "width": video_latent_data["width"],
                "fps": batch["video_metadata"]["fps"][i].item(),
            }
            _atomic_save(latent_data, output_file)

            if with_audio and audio_vae_encoder is not None and audio_processor is not None:
                audio_batch = batch.get("audio")
                if audio_batch is not None:
                    audio_data = encode_audio(
                        audio_vae_encoder,
                        audio_processor,
                        audio_batch["waveform"][i],
                        audio_batch["sample_rate"][i].item(),
                    )
                    audio_output_file = audio_output_path / output_rel_path
                    audio_output_file.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_save(audio_data, audio_output_file)
                    audio_success_count += 1
                else:
                    audio_skip_count += 1

    print(f"Processed {len(dataloader.dataset)} videos -> {output_path}")
    if with_audio:
        print(f"Audio: {audio_success_count} with audio, {audio_skip_count} without (skipped)")


# ---------------------------------------------------------------------------
# Stage 4: Pack precomputed .pt files into parquet
# ---------------------------------------------------------------------------


def save_parquet(
    precomputed_dir: str,
    output_dir: str,
    shard_size: int = 1000,
    pad_to_multiple_of: int | None = None,
) -> None:
    """Pack precomputed ``.pt`` files into parquet shards for offline training.

    Reads ``.pt`` files from ``latents/``, ``conditions/``, and optionally
    ``audio_latents/`` under *precomputed_dir*, merges them per-sample into
    a flat dict, pickles all tensor values, and saves as parquet shards
    compatible with VeOmni's ``process_dit_offline_example`` data transform.

    Output format matches ``OfflineEmbeddingSaver``: each parquet row is a
    dict where every value is ``pickle.dumps(tensor.cpu())`` (bytes).

    Args:
        precomputed_dir: Directory containing latents/, conditions/, and optionally audio_latents/.
        output_dir: Output directory for parquet shards.
        shard_size: Number of samples per parquet shard.
        pad_to_multiple_of: If set, pad total samples to be divisible by this number.
            Useful for ensuring even distribution across DP ranks in distributed training.
            For example, set to ``dp_size`` to prevent FSDP2 deadlocks.
    """
    import pickle as pk

    from datasets import Dataset as HFDataset

    precomputed = Path(precomputed_dir)
    latents_dir = precomputed / "latents"
    conditions_dir = precomputed / "conditions"
    audio_latents_dir = precomputed / "audio_latents"

    if not latents_dir.is_dir():
        print(f"ERROR: latents directory not found: {latents_dir}")
        sys.exit(1)
    if not conditions_dir.is_dir():
        print(f"ERROR: conditions directory not found: {conditions_dir}")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    latent_files = sorted(latents_dir.rglob("*.pt"))
    if not latent_files:
        print(f"No .pt files found in {latents_dir}")
        return

    print(f"Found {len(latent_files)} latent files")

    def _cpu_recursive(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: _cpu_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_cpu_recursive(v) for v in obj)
        return obj

    def _to_bytes(d: dict) -> dict:
        return {k: pk.dumps(_cpu_recursive(v)) for k, v in d.items()}

    all_samples: list[dict] = []
    skipped = 0

    for latent_file in tqdm(latent_files, desc="Packing parquet"):
        rel = latent_file.relative_to(latents_dir)

        cond_file = conditions_dir / rel
        if not cond_file.is_file():
            print(f"  WARNING: condition file missing for {rel}, skipping")
            skipped += 1
            continue

        latent_data = torch.load(latent_file, map_location="cpu", weights_only=True)
        cond_data = torch.load(cond_file, map_location="cpu", weights_only=True)

        merged: dict[str, Any] = {}

        if isinstance(latent_data, dict) and "latents" in latent_data:
            merged["latents"] = latent_data["latents"]
            if "fps" in latent_data:
                merged["fps"] = latent_data["fps"]
        else:
            merged["latents"] = latent_data

        if isinstance(cond_data, dict):
            merged.update(cond_data)
        else:
            merged["conditions"] = cond_data

        if audio_latents_dir.is_dir():
            audio_file = audio_latents_dir / rel
            if audio_file.is_file():
                audio_data = torch.load(audio_file, map_location="cpu", weights_only=True)
                if isinstance(audio_data, dict) and "latents" in audio_data:
                    merged["audio_latents"] = audio_data["latents"]
                    if "num_time_steps" in audio_data:
                        merged["audio_num_time_steps"] = audio_data["num_time_steps"]
                    if "frequency_bins" in audio_data:
                        merged["audio_frequency_bins"] = audio_data["frequency_bins"]
                    if "duration" in audio_data:
                        merged["audio_duration"] = audio_data["duration"]
                else:
                    merged["audio_latents"] = audio_data

        all_samples.append(_to_bytes(merged))

    original_count = len(all_samples)

    if pad_to_multiple_of and pad_to_multiple_of > 1 and all_samples:
        remainder = original_count % pad_to_multiple_of
        if remainder > 0:
            pad_count = pad_to_multiple_of - remainder
            for i in range(pad_count):
                all_samples.append(all_samples[i % original_count])
            print(
                f"Padded {pad_count} samples to make total ({original_count} + {pad_count} = {len(all_samples)}) divisible by {pad_to_multiple_of}"
            )

    shard_index = 0
    total_saved = 0

    for i in range(0, len(all_samples), shard_size):
        chunk = all_samples[i : i + shard_size]
        ds = HFDataset.from_list(chunk)
        ds.to_parquet(str(output_path / f"shard_{shard_index:04d}.parquet"))
        total_saved += len(chunk)
        shard_index += 1

    num_shards = shard_index
    print(f"Packed {total_saved} samples into {num_shards} parquet shards -> {output_path}")
    if skipped:
        print(f"  Skipped {skipped} samples (missing condition files)")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    video_dir: str | None,
    data_dir: str,
    dataset_file: str | None,
    gemma_model_path: str,
    checkpoint_path: str,
    resolution_buckets_str: str,
    caption_column: str = "caption",
    media_column: str = "media_path",
    max_sequence_length: int = 256,
    captioner_type: str = "qwen_omni",
    with_audio: bool = False,
    vae_tiling: bool = False,
    lora_trigger: str | None = None,
    remove_llm_prefixes: bool = False,
    reshape_mode: str = "center",
    reference_column: str | None = None,
    reference_downscale_factor: int = 1,
    device: str | None = None,
    load_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    """Run the full preprocessing pipeline.

    Steps:
    1. Split scenes (if *video_dir* is provided)
    2. Caption videos (if no *dataset_file* is provided)
    3. Compute text embeddings
    4. Compute video latents (+ reference latents for IC-LoRA)
    5. Compute audio latents (if *with_audio*)
    """
    data_path = Path(data_dir)
    precomputed = data_path / ".precomputed"
    conditions_dir = precomputed / "conditions"
    latents_dir = precomputed / "latents"
    audio_latents_dir = precomputed / "audio_latents"

    resolution_buckets = parse_resolution_buckets(resolution_buckets_str)

    if video_dir:
        clips_dir = data_path / "clips"
        split_scenes(video_dir=video_dir, output_dir=str(clips_dir))

        ds_output = data_path / "dataset.json"
        caption_videos(
            input_dir=str(clips_dir),
            output=str(ds_output),
            captioner_type=captioner_type,
            device=device,
        )
        dataset_file = str(ds_output)

    if dataset_file is None:
        candidates = [data_path / "dataset.json", data_path / "dataset.csv", data_path / "dataset.jsonl"]
        for c in candidates:
            if c.exists():
                dataset_file = str(c)
                break
        if dataset_file is None:
            print("ERROR: No dataset file found. Provide --dataset_file or run with --video_dir.")
            sys.exit(1)

    print("\n=== Computing caption embeddings ===")
    compute_caption_embeddings(
        dataset_file=dataset_file,
        output_dir=str(conditions_dir),
        checkpoint_path=checkpoint_path,
        gemma_model_path=gemma_model_path,
        caption_column=caption_column,
        media_column=media_column,
        max_sequence_length=max_sequence_length,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
        device=device,
        load_in_8bit=load_in_8bit,
        overwrite=overwrite,
    )

    print("\n=== Computing video latents ===")
    compute_video_latents(
        dataset_file=dataset_file,
        output_dir=str(latents_dir),
        checkpoint_path=checkpoint_path,
        resolution_buckets=resolution_buckets,
        video_column=media_column,
        reshape_mode=reshape_mode,
        device=device,
        vae_tiling=vae_tiling,
        with_audio=with_audio,
        audio_output_dir=str(audio_latents_dir) if with_audio else None,
        overwrite=overwrite,
    )

    if reference_column:
        if reference_downscale_factor > 1 and len(resolution_buckets) > 1:
            raise ValueError(
                "When using --reference-downscale-factor > 1, only a single resolution bucket is supported."
            )

        reference_buckets = compute_scaled_resolution_buckets(resolution_buckets, reference_downscale_factor)
        reference_latents_dir = precomputed / "reference_latents"

        print(f"\n=== Computing reference video latents (buckets: {reference_buckets}) ===")
        compute_video_latents(
            dataset_file=dataset_file,
            output_dir=str(reference_latents_dir),
            checkpoint_path=checkpoint_path,
            resolution_buckets=reference_buckets,
            video_column=reference_column,
            main_media_column=media_column,
            reshape_mode=reshape_mode,
            device=device,
            vae_tiling=vae_tiling,
            overwrite=overwrite,
        )

    print(f"\nPipeline complete! Results saved to {precomputed}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 preprocessing pipeline: scene splitting, captioning, and latent computation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # --- split-scenes ---
    sp_split = subparsers.add_parser("split-scenes", help="Split videos into scene clips")
    sp_split.add_argument("--video_dir", type=str, required=True, help="Directory containing raw videos")
    sp_split.add_argument("--output_dir", type=str, required=True, help="Output directory for clips")
    sp_split.add_argument("--detector", type=str, default="content", choices=["content", "adaptive", "threshold"])
    sp_split.add_argument("--threshold", type=float, default=None)
    sp_split.add_argument("--min_scene_len", type=int, default=None)
    sp_split.add_argument("--max_scenes", type=int, default=None)
    sp_split.add_argument("--filter_shorter_than", type=str, default=None)
    sp_split.add_argument("--duration", type=str, default=None)
    sp_split.add_argument("--save_images", type=int, default=0)

    # --- caption ---
    sp_caption = subparsers.add_parser("caption", help="Auto-caption videos")
    sp_caption.add_argument("--input_dir", type=str, required=True, help="Directory containing video clips")
    sp_caption.add_argument("--output", type=str, required=True, help="Output dataset file path")
    sp_caption.add_argument("--captioner_type", type=str, default="qwen_omni", choices=["qwen_omni", "gemini_flash"])
    sp_caption.add_argument("--instruction", type=str, default=None)
    sp_caption.add_argument("--fps", type=int, default=3)
    sp_caption.add_argument("--no_audio", action="store_true")
    _add_common_args(sp_caption)

    # --- preprocess ---
    sp_pre = subparsers.add_parser("preprocess", help="Compute text embeddings + VAE latents")
    sp_pre.add_argument("--dataset_file", type=str, required=True, help="Path to dataset CSV/JSON/JSONL")
    sp_pre.add_argument("--checkpoint_path", type=str, required=True, help="Path to LTX-2 checkpoint")
    sp_pre.add_argument("--gemma_model_path", type=str, required=True, help="Path to Gemma3 model")
    sp_pre.add_argument("--resolution_buckets", type=str, required=True, help="WxHxF;WxHxF;...")
    sp_pre.add_argument("--caption_column", type=str, default="caption")
    sp_pre.add_argument("--media_column", type=str, default="media_path")
    sp_pre.add_argument("--max_sequence_length", type=int, default=256)
    sp_pre.add_argument("--lora_trigger", type=str, default=None)
    sp_pre.add_argument("--remove_llm_prefixes", action="store_true")
    sp_pre.add_argument("--reshape_mode", type=str, default="center", choices=["center", "random"])
    sp_pre.add_argument("--with_audio", action="store_true")
    sp_pre.add_argument("--vae_tiling", action="store_true")
    sp_pre.add_argument("--load_in_8bit", action="store_true", help="Load Gemma in 8-bit to save GPU memory")
    sp_pre.add_argument(
        "--reference_column", type=str, default=None, help="Column for reference video paths (IC-LoRA)"
    )
    sp_pre.add_argument(
        "--reference_downscale_factor",
        type=int,
        default=1,
        help="Downscale factor for reference video resolution (IC-LoRA)",
    )
    _add_common_args(sp_pre)

    # --- save-parquet ---
    sp_parquet = subparsers.add_parser("save-parquet", help="Pack precomputed .pt files into parquet shards")
    sp_parquet.add_argument(
        "--precomputed_dir",
        type=str,
        required=True,
        help="Directory containing latents/, conditions/, and optionally audio_latents/",
    )
    sp_parquet.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet shards")
    sp_parquet.add_argument("--shard_size", type=int, default=1000, help="Number of samples per parquet shard")
    sp_parquet.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help="Pad total samples to be divisible by this number (e.g., dp_size for distributed training)",
    )

    # --- all ---
    sp_all = subparsers.add_parser("all", help="Run full pipeline: split → caption → preprocess")
    sp_all.add_argument("--video_dir", type=str, default=None, help="Raw video directory (optional)")
    sp_all.add_argument("--data_dir", type=str, required=True, help="Output data directory")
    sp_all.add_argument("--dataset_file", type=str, default=None, help="Existing dataset file (skip split+caption)")
    sp_all.add_argument("--checkpoint_path", type=str, required=True, help="Path to LTX-2 checkpoint")
    sp_all.add_argument("--gemma_model_path", type=str, required=True, help="Path to Gemma3 model")
    sp_all.add_argument("--resolution_buckets", type=str, required=True, help="WxHxF;WxHxF;...")
    sp_all.add_argument("--caption_column", type=str, default="caption")
    sp_all.add_argument("--media_column", type=str, default="media_path")
    sp_all.add_argument("--max_sequence_length", type=int, default=256)
    sp_all.add_argument("--captioner_type", type=str, default="qwen_omni")
    sp_all.add_argument("--lora_trigger", type=str, default=None)
    sp_all.add_argument("--remove_llm_prefixes", action="store_true")
    sp_all.add_argument("--reshape_mode", type=str, default="center", choices=["center", "random"])
    sp_all.add_argument("--with_audio", action="store_true")
    sp_all.add_argument("--vae_tiling", action="store_true")
    sp_all.add_argument("--load_in_8bit", action="store_true")
    sp_all.add_argument("--reference_column", type=str, default=None)
    sp_all.add_argument("--reference_downscale_factor", type=int, default=1)
    _add_common_args(sp_all)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "split-scenes":
        split_scenes(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            detector=args.detector,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            max_scenes=args.max_scenes,
            filter_shorter_than=args.filter_shorter_than,
            duration=args.duration,
            save_images=args.save_images,
        )

    elif args.command == "caption":
        caption_videos(
            input_dir=args.input_dir,
            output=args.output,
            captioner_type=args.captioner_type,
            device=args.device,
            instruction=args.instruction,
            fps=args.fps,
            include_audio=not args.no_audio,
        )

    elif args.command == "preprocess":
        buckets = parse_resolution_buckets(args.resolution_buckets)

        dataset_path = Path(args.dataset_file)
        precomputed = dataset_path.parent / ".precomputed"

        compute_caption_embeddings(
            dataset_file=args.dataset_file,
            output_dir=str(precomputed / "conditions"),
            checkpoint_path=args.checkpoint_path,
            gemma_model_path=args.gemma_model_path,
            caption_column=args.caption_column,
            media_column=args.media_column,
            max_sequence_length=args.max_sequence_length,
            lora_trigger=args.lora_trigger,
            remove_llm_prefixes=args.remove_llm_prefixes,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            overwrite=args.overwrite,
        )

        compute_video_latents(
            dataset_file=args.dataset_file,
            output_dir=str(precomputed / "latents"),
            checkpoint_path=args.checkpoint_path,
            resolution_buckets=buckets,
            video_column=args.media_column,
            reshape_mode=args.reshape_mode,
            device=args.device,
            vae_tiling=args.vae_tiling,
            with_audio=args.with_audio,
            audio_output_dir=str(precomputed / "audio_latents") if args.with_audio else None,
            overwrite=args.overwrite,
        )

        if args.reference_column:
            if args.reference_downscale_factor > 1 and len(buckets) > 1:
                print("ERROR: --reference-downscale-factor > 1 requires a single resolution bucket.")
                sys.exit(1)

            ref_buckets = compute_scaled_resolution_buckets(buckets, args.reference_downscale_factor)
            reference_latents_dir = precomputed / "reference_latents"

            compute_video_latents(
                dataset_file=args.dataset_file,
                output_dir=str(reference_latents_dir),
                checkpoint_path=args.checkpoint_path,
                resolution_buckets=ref_buckets,
                video_column=args.reference_column,
                main_media_column=args.media_column,
                reshape_mode=args.reshape_mode,
                device=args.device,
                vae_tiling=args.vae_tiling,
                overwrite=args.overwrite,
            )

    elif args.command == "save-parquet":
        save_parquet(
            precomputed_dir=args.precomputed_dir,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            pad_to_multiple_of=args.pad_to_multiple_of,
        )

    elif args.command == "all":
        run_pipeline(
            video_dir=args.video_dir,
            data_dir=args.data_dir,
            dataset_file=args.dataset_file,
            gemma_model_path=args.gemma_model_path,
            checkpoint_path=args.checkpoint_path,
            resolution_buckets_str=args.resolution_buckets,
            caption_column=args.caption_column,
            media_column=args.media_column,
            max_sequence_length=args.max_sequence_length,
            captioner_type=args.captioner_type,
            with_audio=args.with_audio,
            vae_tiling=args.vae_tiling,
            lora_trigger=args.lora_trigger,
            remove_llm_prefixes=args.remove_llm_prefixes,
            reshape_mode=args.reshape_mode,
            reference_column=args.reference_column,
            reference_downscale_factor=args.reference_downscale_factor,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
