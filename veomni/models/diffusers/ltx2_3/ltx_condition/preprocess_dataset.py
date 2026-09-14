"""End-to-end preprocessing pipeline for LTX-2 offline training.

Integrates scene splitting, video captioning, text embedding computation,
and video/audio latent encoding into a single script.

Subcommands:
    split-scenes       Split raw videos into scene clips using PySceneDetect
    caption            Auto-caption video clips using a multimodal model
    compute-reference  Generate Canny edge reference videos for IC-LoRA
    preprocess         Compute text embeddings + VAE latents from a dataset file
    save-parquet       Pack precomputed .pt files into parquet for offline training

Usage examples:

    # Split raw videos into scene clips using PySceneDetect
    python preprocess_dataset.py split-scenes \
        --video_dir /path/to/raw/videos \
        --output_dir /path/to/clips

    # Split with custom detector and filter short scenes
    python preprocess_dataset.py split-scenes \
        --video_dir /path/to/raw/videos \
        --output_dir /path/to/clips \
        --detector adaptive \
        --filter_shorter_than 2s \
        --max_scenes 10 \
        --save_images 3

    # Generate dataset.json by auto-captioning videos (Qwen2.5-Omni, local)
    python preprocess_dataset.py caption \
        --input_dir /path/to/videos \
        --output /path/to/videos/dataset.json

    # Generate dataset.json with Gemini API (requires GOOGLE_API_KEY env var)
    python preprocess_dataset.py caption \
        --input_dir /path/to/videos \
        --output /path/to/videos/dataset.json \
        --captioner_type gemini_flash

    # Generate dataset.json without audio processing
    python preprocess_dataset.py caption \
        --input_dir /path/to/videos \
        --output /path/to/videos/dataset.json \
        --no_audio

    # Generate Canny edge reference videos for IC-LoRA
    # (reads dataset.json, generates *_reference.mp4, updates dataset.json in-place)
    python preprocess_dataset.py compute-reference \
        --input_dir /path/to/videos \
        --dataset_file /path/to/videos/dataset.json

    # Only preprocess (text embeddings + VAE latents)
    python preprocess_dataset.py preprocess \
        --dataset_file /path/to/dataset.json \
        --gemma_model_path /path/to/gemma3 \
        --checkpoint_path /path/to/ltx2.safetensors \
        --resolution_buckets 768x768x49

    # Preprocess with reference videos for IC-LoRA
    # (auto-generates reference videos before encoding latents)
    python preprocess_dataset.py preprocess \
        --dataset_file /path/to/dataset.json \
        --gemma_model_path /path/to/gemma3 \
        --checkpoint_path /path/to/ltx2.safetensors \
        --resolution_buckets 768x768x49 \
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
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import veomni.models.diffusers.ltx2_3.ltx_core  # noqa: F401
from veomni.utils.device import get_device_type


register_heif_opener()

VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8
DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 128
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "heif", "heic"]


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
    stats_file: str | None = None,
    luma_only: bool = False,
    adaptive_window: int | None = None,
    fade_bias: float | None = None,
    downscale_factor: int | None = None,
    frame_skip: int = 0,
) -> list[Path]:
    """Split all videos in *video_dir* into scene clips saved under *output_dir*.

    Requires ``scenedetect`` and ``ffmpeg`` to be installed.
    Returns a list of paths to the generated clip files.
    """
    try:
        from scenedetect import (
            AdaptiveDetector,
            ContentDetector,
            HistogramDetector,
            SceneManager,
            ThresholdDetector,
            open_video,
        )
        from scenedetect.scene_manager import save_images as save_scene_images
        from scenedetect.scene_manager import write_scene_list_html
        from scenedetect.stats_manager import StatsManager
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

        stats_manager = StatsManager() if stats_file else None
        scene_manager = SceneManager(stats_manager)

        if downscale_factor:
            scene_manager.auto_downscale = False
            scene_manager.downscale = downscale_factor

        kwargs: dict[str, Any] = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if min_scene_len is not None:
            kwargs["min_scene_len"] = min_scene_len

        if detector == "content":
            if luma_only:
                kwargs["luma_only"] = luma_only
            scene_manager.add_detector(ContentDetector(**kwargs))
        elif detector == "adaptive":
            if adaptive_window is not None:
                kwargs["window_width"] = adaptive_window
            if luma_only:
                kwargs["luma_only"] = luma_only
            if "threshold" in kwargs:
                kwargs["adaptive_threshold"] = kwargs.pop("threshold")
            scene_manager.add_detector(AdaptiveDetector(**kwargs))
        elif detector == "threshold":
            if fade_bias is not None:
                kwargs["fade_bias"] = fade_bias
            scene_manager.add_detector(ThresholdDetector(**kwargs))
        elif detector == "histogram":
            scene_manager.add_detector(HistogramDetector(**kwargs))
        else:
            scene_manager.add_detector(ContentDetector(**kwargs))

        scene_manager.detect_scenes(video=video, show_progress=True, frame_skip=frame_skip, duration=duration_tc)
        scenes = scene_manager.get_scene_list()

        if filter_tc:
            original_count = len(scenes)
            scenes = [(s, e) for s, e in scenes if (e.get_frames() - s.get_frames()) >= filter_tc.get_frames()]
            if len(scenes) < original_count:
                print(
                    f"  Filtered out {original_count - len(scenes)} scenes shorter "
                    f"than {filter_tc.get_seconds():.1f} seconds ({filter_tc.get_frames()} frames)"
                )

        if max_scenes and len(scenes) > max_scenes:
            print(f"  Dropping last {len(scenes) - max_scenes} scenes to meet max_scenes ({max_scenes}) limit")
            scenes = scenes[:max_scenes]

        print(f"  {video_file.name}: {len(scenes)} scenes detected")

        if stats_file:
            print(f"  Saving detection stats to {stats_file}")
            stats_manager.save_to_csv(stats_file)

        split_video_ffmpeg(
            input_video_path=str(video_file),
            scene_list=scenes,
            output_dir=str(clip_dir),
            show_progress=True,
        )

        if save_images > 0:
            image_filenames = save_scene_images(
                scene_list=scenes,
                video=video,
                num_images=save_images,
                output_dir=str(clip_dir),
                show_progress=True,
            )

            html_path = clip_dir / "scene_report.html"
            write_scene_list_html(
                output_html_filename=str(html_path),
                scene_list=scenes,
                image_filenames=image_filenames,
            )
            print(f"  Scene report saved to: {html_path}")

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
    - ``qwen_omni``: local Qwen2.5-Omni model (bundled in ``captioning.py``)
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
    from veomni.models.diffusers.ltx2_3.ltx_condition.captioning import CaptionerType, create_captioner

    device_str = device or get_device_type()
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


COMMON_BEGINNING_PHRASES: tuple[str, ...] = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)

COMMON_CONTINUATION_WORDS: tuple[str, ...] = (
    "shows",
    "depicts",
    "features",
    "captures",
    "highlights",
    "introduces",
    "presents",
)

COMMON_LLM_START_PHRASES: tuple[str, ...] = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in COMMON_BEGINNING_PHRASES
        for continuation in COMMON_CONTINUATION_WORDS
    ),
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
    """Load feature extractor using EmbeddingsProcessor (matches LTX-2 exactly)."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        EmbeddingsProcessorConfigurator,
    )
    from ltx_core.utils import find_matching_file

    checkpoint_file = find_matching_file(str(checkpoint_path), "*.safetensors")
    embeddings_processor = SingleGPUModelBuilder(
        model_path=str(checkpoint_file),
        model_class_configurator=EmbeddingsProcessorConfigurator,
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    ).build(device=device, dtype=dtype)

    return embeddings_processor.feature_extractor


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
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        GEMMA_LLM_KEY_OPS,
        GEMMA_MODEL_OPS,
        GemmaTextEncoderConfigurator,
        module_ops_from_gemma_root,
    )
    from ltx_core.utils import find_matching_file

    gemma_model_folder = find_matching_file(str(gemma_model_path), "model*.safetensors").parent
    gemma_weight_paths = [str(p) for p in gemma_model_folder.rglob("*.safetensors")]

    text_encoder = SingleGPUModelBuilder(
        model_path=tuple(gemma_weight_paths),
        model_class_configurator=GemmaTextEncoderConfigurator,
        model_sd_ops=GEMMA_LLM_KEY_OPS,
        module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(str(gemma_model_path))),
    ).build(device=dev, dtype=dtype)
    text_encoder.eval()

    print("Loading feature extractor from checkpoint...")
    feature_extractor = load_feature_extractor(checkpoint_path, dev, dtype)
    feature_extractor.eval()

    print(f"Processing captions in {len(dataloader):,} batches...")

    for batch in tqdm(dataloader, desc="Encoding captions"):
        with torch.inference_mode():
            for i in range(len(batch["prompt"])):
                hidden_states, prompt_attention_mask = text_encoder.encode(batch["prompt"][i], padding_side="left")
                video_feats, audio_feats = feature_extractor(hidden_states, prompt_attention_mask, "left")

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
    """Read video frames as ``[F, C, H, W]`` float tensor in ``[0, 1]`` using PyAV."""
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate or video_stream.base_rate or 24)

        frames = []
        for frame in container.decode(video=0):
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    frames_np = np.stack(frames, axis=0)
    video = torch.from_numpy(frames_np).float().div(255.0)
    return video.permute(0, 3, 1, 2), fps


def _get_video_frame_count(video_path: Path) -> int:
    """Get frame count using PyAV (matches LTX-2 exactly)."""
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]

        if video_stream.frames > 0:
            return video_stream.frames

        rate = video_stream.average_rate or video_stream.base_rate
        if video_stream.duration and video_stream.time_base and rate:
            duration = Fraction(video_stream.duration) * Fraction(video_stream.time_base)
            return round(duration * Fraction(rate))

        return sum(1 for _ in container.decode(video=0))


def _open_image_as_srgb(image_path):
    """Open image with EXIF rotation and ICC profile conversion (matches LTX-2)."""
    import io

    from PIL import ExifTags, ImageCms

    exif_colorspace_srgb = 1

    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        exif_data = img.getexif()
        if exif_data is not None:
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            if color_space_value is not None and color_space_value != exif_colorspace_srgb:
                raise ValueError(
                    f"Image has colorspace tag in EXIF but it isn't set to sRGB, "
                    f"conversion is not supported. EXIF ColorSpace tag value is {color_space_value}"
                )

        srgb_img = img.convert("RGB")
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img


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
    """Dataset for processing video and image files with resolution bucket selection.

    Loads videos/images from CSV/JSON/JSONL metadata, applies resize/crop transforms,
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

        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.clamp_(0, 1)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, list):
            return index

        video_path: Path = self.video_paths[index]
        data_root = self.dataset_file.parent
        relative_path = str(video_path.relative_to(data_root))
        media_relative_path = str(self.main_media_paths[index].relative_to(data_root))

        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_tensor = self._preprocess_image(video_path)
            fps = 1.0
            audio_data = None
        else:
            media_tensor, fps = self._preprocess_video(video_path)

            if self.with_audio:
                target_duration = media_tensor.shape[1] / fps
                audio_data = self._extract_audio(video_path, target_duration)
            else:
                audio_data = None

        _, num_frames, height, width = media_tensor.shape

        result: dict[str, Any] = {
            "video": media_tensor,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": fps,
            },
        }

        if audio_data is not None:
            result["audio"] = audio_data

        return result

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image (matches LTX-2 exactly)."""
        image = _open_image_as_srgb(path)
        image = to_tensor(image)
        image = image.unsqueeze(0)

        nearest_bucket = self._get_resolution_bucket(image)
        _, target_height, target_width = nearest_bucket
        image_resized = self._resize_and_crop(image, target_height, target_width)

        image = self.transforms(image_resized)
        image = image.unsqueeze(1)
        return image

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess a video (matches LTX-2 exactly)."""
        video, fps = _read_video_frames(path, self.max_target_frames)

        nearest_bucket = self._get_resolution_bucket(video)
        target_num_frames, target_height, target_width = nearest_bucket
        frames_resized = self._resize_and_crop(video, target_height, target_width)

        frames_resized = frames_resized[:target_num_frames]

        video = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)
        video = video.permute(1, 0, 2, 3).contiguous()

        return video, fps

    def _resize_and_crop(self, media_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize and crop tensor to target size (matches LTX-2 exactly)."""
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import crop, resize

        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]

        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        if current_aspect > target_aspect:
            new_width = int(current_width * target_height / current_height)
            media_tensor = resize(
                media_tensor,
                size=[target_height, new_width],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            new_height = int(current_height * target_width / current_width)
            media_tensor = resize(
                media_tensor,
                size=[new_height, target_width],
                interpolation=InterpolationMode.BICUBIC,
            )

        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]
        media_tensor = media_tensor.squeeze(0)

        delta_h = current_height - target_height
        delta_w = current_width - target_width

        if self.reshape_mode == "random":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif self.reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unsupported reshape mode: {self.reshape_mode}")

        media_tensor = crop(media_tensor, top=top, left=left, height=target_height, width=target_width)
        return media_tensor

    def _get_resolution_bucket(self, media_tensor: torch.Tensor) -> tuple[int, int, int]:
        """Get the nearest resolution bucket for the given media tensor."""
        num_frames, _, height, width = media_tensor.shape

        def distance(bucket: tuple[int, int, int]) -> tuple:
            bf, bh, bw = bucket
            return (
                abs(math.log(width / height) - math.log(bw / bh)),
                -bf,
                -(bh * bw),
            )

        relevant = [b for b in self.resolution_buckets if b[0] <= num_frames]
        if not relevant:
            raise ValueError(f"No bucket has <= {num_frames} frames. Buckets: {self.resolution_buckets}")

        return min(relevant, key=distance)

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

            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_video_paths.append(video_path)
                valid_main_media_paths.append(self.main_media_paths[i])
                continue

            try:
                frame_count = _get_video_frame_count(video_path)
                if frame_count >= min_frames_required:
                    valid_video_paths.append(video_path)
                    valid_main_media_paths.append(self.main_media_paths[i])
                else:
                    print(f"  Skipping {video_path} — {frame_count} frames < {min_frames_required}")
            except Exception:
                pass

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

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    duration = waveform.shape[-1] / sampling_rate

    mel_device = device if device.type == get_device_type() else "cpu"
    waveform_mel = waveform.to(device=mel_device, dtype=dtype)
    mel = audio_processor.waveform_to_mel(Audio(waveform=waveform_mel, sampling_rate=sampling_rate))
    mel = mel.to(device=device, dtype=dtype)

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
        from ltx_core.loader import SingleGPUModelBuilder
        from ltx_core.model.audio_vae import (
            AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            AudioEncoderConfigurator,
            AudioProcessor,
        )

        def load_audio_encoder(
            checkpoint_path: str | Path,
            device: str | torch.device = "cpu",
            dtype: torch.dtype = torch.bfloat16,
        ):
            if isinstance(device, str):
                device = torch.device(device)
            return SingleGPUModelBuilder(
                model_path=str(checkpoint_path),
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            ).build(device=device, dtype=dtype)

        print("Loading audio VAE encoder...")
        audio_vae_encoder = load_audio_encoder(checkpoint_path, device=dev, dtype=torch.float32)
        audio_vae_encoder.eval()

        mel_device = dev if dev.type == get_device_type() else torch.device("cpu")
        audio_processor = AudioProcessor(
            target_sample_rate=audio_vae_encoder.sample_rate,
            mel_bins=audio_vae_encoder.mel_bins,
            mel_hop_length=audio_vae_encoder.mel_hop_length,
            n_fft=audio_vae_encoder.n_fft,
        ).to(mel_device)

    audio_success_count = 0
    audio_skip_count = 0

    for batch in tqdm(dataloader, desc="Encoding videos"):
        video = batch["video"]

        with torch.inference_mode():
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
# Stage 3c: Compute reference videos for IC-LoRA training
# ---------------------------------------------------------------------------


def compute_reference_frames(images: torch.Tensor) -> torch.Tensor:
    """Compute Canny edge detection on a batch of images.

    Args:
        images: Batch of images tensor of shape ``[B, C, H, W]`` in ``[0, 1]``.

    Returns:
        Binary edge masks tensor of shape ``[B, 3, H, W]``.
    """
    if images.shape[1] == 3:
        images = TF.rgb_to_grayscale(images)

    if images.max() > 1.0:
        images = images / 255.0

    edge_masks = []
    for image in images:
        image_np = (image.squeeze().cpu().numpy() * 255).astype("uint8")
        edges = cv2.Canny(image_np, threshold1=100, threshold2=200)
        edge_mask = torch.from_numpy(edges).float()
        edge_masks.append(edge_mask)

    edges = torch.stack(edge_masks)
    edges = torch.stack([edges] * 3, dim=1)
    return edges


def save_reference_video(video: torch.Tensor, output_path: Path, fps: float) -> None:
    """Save edge-detected frames as a reference video using PyAV.

    Args:
        video: ``[F, C, H, W]`` float tensor in ``[0, 255]``.
        output_path: Output video file path.
        fps: Frames per second for the output video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames, _channels, height, width = video.shape

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=round(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}

    for i in range(num_frames):
        frame_np = video[i].permute(1, 2, 0).cpu().numpy().astype("uint8")
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def compute_reference_videos(
    input_dir: str,
    dataset_file: str,
    override: bool = False,
    batch_size: int = 100,
) -> None:
    """Generate Canny edge reference videos for IC-LoRA training.

    Reads videos listed in *dataset_file*, generates Canny edge detection
    reference videos (named ``<stem>_reference.<ext>``), and updates
    *dataset_file* in-place to add a ``reference_path`` field to each entry.

    Args:
        input_dir: Base directory for resolving video paths.
        dataset_file: Path to dataset JSON file (must already exist).
        override: Whether to regenerate existing reference videos.
        batch_size: Number of frames to process per batch.
    """
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    base_dir = Path(input_dir).resolve()

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    def _media_path_to_reference_path(media_file: Path) -> Path:
        return media_file.parent / (media_file.stem + "_reference" + media_file.suffix)

    media_paths = [base_dir / Path(entry["media_path"]) for entry in dataset]
    reference_paths = {
        entry["media_path"]: str(
            _media_path_to_reference_path(base_dir / Path(entry["media_path"])).relative_to(base_dir)
        )
        for entry in dataset
    }

    processed = 0
    for media_file in tqdm(media_paths, desc="Computing reference videos"):
        rel_path = str(media_file.resolve().relative_to(base_dir))
        reference_path = _media_path_to_reference_path(media_file)
        reference_paths[rel_path] = str(reference_path.relative_to(base_dir))

        if not reference_path.resolve().exists() or override:
            try:
                video, fps = _read_video_frames(media_file, max_frames=999999)

                condition_frames = []
                for i in range(0, len(video), batch_size):
                    batch = video[i : i + batch_size]
                    condition_batch = compute_reference_frames(batch)
                    condition_frames.append(condition_batch)

                all_condition = torch.cat(condition_frames, dim=0)
                save_reference_video(all_condition, reference_path.resolve(), fps=fps)
                processed += 1

            except Exception as e:
                print(f"  WARNING: Error processing {media_file}: {e}")
                reference_paths.pop(rel_path, None)

    for entry in dataset:
        entry["reference_path"] = reference_paths[entry["media_path"]]

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Processed {processed}/{len(media_paths)} reference videos")
    print(f"Updated dataset file: {dataset_path}")


# ---------------------------------------------------------------------------
# Stage 4: Pack precomputed .pt files into parquet
# ---------------------------------------------------------------------------


def save_parquet(
    precomputed_dir: str,
    output_dir: str,
    shard_size: int = 1000,
    pad_to_multiple_of: int | None = None,
    with_audio: bool = False,
    reference_latents_dir: str | None = None,
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
        with_audio: If True, validate and report audio field coverage.
            Warns when audio_latents or audio_prompt_embeds are missing.
    """
    import pickle as pk

    from datasets import Dataset as HFDataset

    precomputed = Path(precomputed_dir)
    latents_dir = precomputed / "latents"
    conditions_dir = precomputed / "conditions"
    audio_latents_dir = precomputed / "audio_latents"
    ref_latents_dir = (
        precomputed / reference_latents_dir if reference_latents_dir else precomputed / "reference_latents"
    )

    if not latents_dir.is_dir():
        print(f"ERROR: latents directory not found: {latents_dir}")
        sys.exit(1)
    if not conditions_dir.is_dir():
        print(f"ERROR: conditions directory not found: {conditions_dir}")
        sys.exit(1)

    has_audio_latents_dir = audio_latents_dir.is_dir()
    if with_audio and not has_audio_latents_dir:
        print(
            f"WARNING: --with_audio is set but audio_latents directory not found: {audio_latents_dir}\n"
            f"  Audio latents will NOT be included in the parquet output.\n"
            f"  Re-run the 'preprocess' command with --with_audio to generate audio latents."
        )

    has_ref_latents_dir = ref_latents_dir.is_dir()
    if reference_latents_dir and not has_ref_latents_dir:
        print(f"WARNING: reference_latents directory not found: {ref_latents_dir}")

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
    audio_latents_count = 0
    audio_prompt_embeds_count = 0
    missing_audio_latents = 0
    missing_audio_prompt_embeds = 0

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

        if "audio_prompt_embeds" in merged:
            audio_prompt_embeds_count += 1
        elif with_audio:
            missing_audio_prompt_embeds += 1

        if has_audio_latents_dir:
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
                audio_latents_count += 1
            elif with_audio:
                missing_audio_latents += 1

        if has_ref_latents_dir:
            ref_file = ref_latents_dir / rel
            if ref_file.is_file():
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=True)
                if isinstance(ref_data, dict) and "latents" in ref_data:
                    merged["reference_latents"] = ref_data["latents"]
                    if "num_frames" in ref_data:
                        merged["reference_num_frames"] = ref_data["num_frames"]
                    if "height" in ref_data:
                        merged["reference_height"] = ref_data["height"]
                    if "width" in ref_data:
                        merged["reference_width"] = ref_data["width"]
                else:
                    merged["reference_latents"] = ref_data

        if len(all_samples) == 0:
            _audio_keys = [k for k in merged if "audio" in k]
            print(f"  First sample ({rel}) fields: {sorted(merged.keys())}")
            if with_audio:
                print(f"  First sample audio fields: {_audio_keys if _audio_keys else '(none)'}")

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

    if with_audio or audio_latents_count > 0 or audio_prompt_embeds_count > 0:
        print("\n=== Audio field summary ===")
        print(f"  audio_latents:       {audio_latents_count}/{original_count} samples")
        print(f"  audio_prompt_embeds: {audio_prompt_embeds_count}/{original_count} samples")
        if with_audio:
            _audio_ok = audio_latents_count > 0 and audio_prompt_embeds_count > 0
            if _audio_ok:
                print("  Audio-video training is READY.")
            else:
                _missing_parts = []
                if missing_audio_latents > 0:
                    _missing_parts.append(f"audio_latents missing for {missing_audio_latents} samples")
                if missing_audio_prompt_embeds > 0:
                    _missing_parts.append(f"audio_prompt_embeds missing for {missing_audio_prompt_embeds} samples")
                if audio_latents_count == 0 and not has_audio_latents_dir:
                    _missing_parts.append("audio_latents/ directory does not exist")
                if audio_prompt_embeds_count == 0:
                    _missing_parts.append("conditions files lack audio_prompt_embeds")
                print(
                    f"  WARNING: Audio data is INCOMPLETE — loss will be video-only.\n"
                    f"  Issues: {'; '.join(_missing_parts)}\n"
                    f"  Fix: re-run 'preprocess' with --with_audio, then re-run 'save-parquet --with_audio'."
                )


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
    sp_split.add_argument(
        "--detector", type=str, default="content", choices=["content", "adaptive", "threshold", "histogram"]
    )
    sp_split.add_argument("--threshold", type=float, default=None)
    sp_split.add_argument("--min_scene_len", type=int, default=None)
    sp_split.add_argument("--max_scenes", type=int, default=None)
    sp_split.add_argument("--filter_shorter_than", type=str, default=None)
    sp_split.add_argument("--duration", type=str, default=None)
    sp_split.add_argument("--save_images", type=int, default=0)
    sp_split.add_argument("--stats_file", type=str, default=None, help="Path to save detection statistics CSV")
    sp_split.add_argument("--luma_only", action="store_true", help="Only use brightness for content detection")
    sp_split.add_argument("--adaptive_window", type=int, default=None, help="Window size for adaptive detection")
    sp_split.add_argument("--fade_bias", type=float, default=None, help="Bias for fade detection (-1.0 to 1.0)")
    sp_split.add_argument(
        "--downscale_factor", type=int, default=None, help="Factor to downscale frames during detection"
    )
    sp_split.add_argument("--frame_skip", type=int, default=0, help="Number of frames to skip during processing")

    # --- caption ---
    sp_caption = subparsers.add_parser("caption", help="Auto-caption videos")
    sp_caption.add_argument("--input_dir", type=str, required=True, help="Directory containing video clips")
    sp_caption.add_argument("--output", type=str, required=True, help="Output dataset file path")
    sp_caption.add_argument("--captioner_type", type=str, default="qwen_omni", choices=["qwen_omni", "gemini_flash"])
    sp_caption.add_argument("--instruction", type=str, default=None)
    sp_caption.add_argument("--fps", type=int, default=3)
    sp_caption.add_argument("--no_audio", action="store_true")
    _add_common_args(sp_caption)

    # --- compute-reference ---
    sp_ref = subparsers.add_parser("compute-reference", help="Compute Canny edge reference videos for IC-LoRA")
    sp_ref.add_argument("--input_dir", type=str, required=True, help="Base directory for resolving video paths")
    sp_ref.add_argument("--dataset_file", type=str, required=True, help="Path to dataset JSON file")
    sp_ref.add_argument("--batch_size", type=int, default=100, help="Batch size for processing frames")
    sp_ref.add_argument("--override", action="store_true", help="Override existing reference video files")
    _add_common_args(sp_ref)

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
    sp_parquet.add_argument(
        "--with_audio",
        action="store_true",
        help="Validate and report audio field coverage (audio_latents + audio_prompt_embeds)",
    )
    sp_parquet.add_argument(
        "--reference_latents_dir",
        type=str,
        default=None,
        help="Subdirectory name for reference latents (for IC-LoRA training)",
    )

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
            stats_file=args.stats_file,
            luma_only=args.luma_only,
            adaptive_window=args.adaptive_window,
            fade_bias=args.fade_bias,
            downscale_factor=args.downscale_factor,
            frame_skip=args.frame_skip,
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

    elif args.command == "compute-reference":
        compute_reference_videos(
            input_dir=args.input_dir,
            dataset_file=args.dataset_file,
            override=args.override,
            batch_size=args.batch_size,
        )

    elif args.command == "preprocess":
        if args.reference_column:
            print("\n=== Computing reference videos ===")
            compute_reference_videos(
                input_dir=str(Path(args.dataset_file).parent),
                dataset_file=args.dataset_file,
                override=args.overwrite,
            )

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
            with_audio=args.with_audio,
            reference_latents_dir=args.reference_latents_dir,
        )


if __name__ == "__main__":
    main()
