"""
Audio-visual media captioning using multimodal models.
This module provides captioning capabilities for videos with audio using:
- Qwen2.5-Omni: Local model supporting text, audio, image, and video inputs (default)
- Gemini Flash: Cloud-based API for audio-visual captioning
Requirements:
- Qwen2.5-Omni: transformers>=4.50, torch
- Gemini Flash: google-generativeai (uv pip install google-generativeai)
  Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable

Copied from ltx-trainer (https://github.com/Lightricks/LTX-Video).
"""

import itertools
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch


DEFAULT_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[SPEECH]: <Word-for-word transcription of everything spoken.
           Listen carefully and transcribe the exact words. If no speech, write "None">
[SOUNDS]: <Description of music, ambient sounds, sound effects. If none, write "None">
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in all four sections. For [SPEECH], transcribe the actual words spoken, not a summary."""

VIDEO_ONLY_CAPTION_INSTRUCTION = """\
Analyze this media and provide a detailed caption in the following EXACT format. Fill in ALL sections:

[VISUAL]: <Detailed description of people, objects, actions, settings, colors, and movements>
[TEXT]: <Any on-screen text visible. If none, write "None">

You MUST fill in both sections."""


class CaptionerType(str, Enum):
    """Enum for different types of media captioners."""

    QWEN_OMNI = "qwen_omni"
    GEMINI_FLASH = "gemini_flash"


def create_captioner(captioner_type: CaptionerType, **kwargs) -> "MediaCaptioningModel":
    """Factory function to create a media captioner."""
    match captioner_type:
        case CaptionerType.QWEN_OMNI:
            return QwenOmniCaptioner(**kwargs)
        case CaptionerType.GEMINI_FLASH:
            return GeminiFlashCaptioner(**kwargs)
        case _:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")


class MediaCaptioningModel(ABC):
    """Abstract base class for audio-visual media captioning models."""

    @abstractmethod
    def caption(self, path: str | Path, **kwargs) -> str:
        """Generate a caption for the given video or image."""

    @property
    @abstractmethod
    def supports_audio(self) -> bool:
        """Whether this captioner supports audio input."""

    @staticmethod
    def _is_image_file(path: str | Path) -> bool:
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _is_video_file(path: str | Path) -> bool:
        return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    @staticmethod
    def _clean_raw_caption(caption: str) -> str:
        start = ["The", "This"]
        kind = ["video", "image", "scene", "animated sequence", "clip", "footage"]
        act = ["displays", "shows", "features", "depicts", "presents", "showcases", "captures", "contains"]

        for x, y, z in itertools.product(start, kind, act):
            caption = caption.replace(f"{x} {y} {z} ", "", 1)

        return caption


class QwenOmniCaptioner(MediaCaptioningModel):
    MODEL_ID = os.environ.get("QWEN_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")

    DEFAULT_SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(
        self,
        device: str | torch.device | None = None,
        use_8bit: bool = False,
        instruction: str | None = None,
    ):
        from veomni.utils.device import get_device_type

        self.device = torch.device(device or get_device_type())
        self.instruction = instruction
        self._load_model(use_8bit=use_8bit)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 1,
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        path = Path(path)
        is_image = self._is_image_file(path)
        is_video = self._is_video_file(path)

        use_audio = include_audio and is_video

        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        user_content = []

        if is_image:
            user_content.append({"type": "image", "image": str(path)})
        elif is_video:
            user_content.append({"type": "video", "video": str(path)})

        user_content.append({"type": "text", "text": instruction})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.DEFAULT_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            load_audio_from_video=use_audio,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
            padding=True,
            use_audio_in_video=use_audio,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        output_tokens = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio,
            do_sample=False,
            max_new_tokens=1024,
        )

        generated_tokens = output_tokens[:, input_len:]

        caption_raw = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        caption_raw = re.split(r"\nHuman(?::|(?:\s*\n)|$)", caption_raw, maxsplit=1)[0]
        caption_raw = caption_raw.strip()

        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _load_model(self, use_8bit: bool) -> None:
        from transformers import (  # noqa: PLC0415
            BitsAndBytesConfig,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

        is_npu = self.device.type == "npu"
        if is_npu:
            import torch_npu  # noqa: PLC0415, F401

            torch.npu.set_device(self.device.index or 0)
            dtype = torch.float16
            device_map = {"": f"npu:{self.device.index or 0}"}
        else:
            dtype = torch.bfloat16
            device_map = "auto"

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            device_map=device_map,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.MODEL_ID)


class GeminiFlashCaptioner(MediaCaptioningModel):
    MODEL_ID = "gemini-flash-lite-latest"

    def __init__(
        self,
        api_key: str | None = None,
        instruction: str | None = None,
    ):
        self.instruction = instruction
        self._init_client(api_key)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 3,  # noqa: ARG002
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        import time  # noqa: PLC0415

        path = Path(path)
        is_video = self._is_video_file(path)
        use_audio = include_audio and is_video

        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        uploaded_file = self._genai.upload_file(path)

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = self._genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

        response = self._model.generate_content([uploaded_file, instruction])

        caption_raw = response.text

        self._genai.delete_file(uploaded_file.name)

        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _init_client(self, api_key: str | None) -> None:
        import os  # noqa: PLC0415

        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "The `google-generativeai` package is required for Gemini Flash captioning. "
                "Install it with: `uv pip install google-generativeai`"
            ) from e

        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required. Provide it via the `api_key` argument "
                "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=resolved_api_key)

        self._genai = genai

        self._model = genai.GenerativeModel(self.MODEL_ID)
