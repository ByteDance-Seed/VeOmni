# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
VeOmni customized Qwen3OmniMoe processor.

This module extends the HuggingFace Qwen3OmniMoe processor with VeOmni-specific modifications.
"""

from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeImagesKwargs,
    Qwen3OmniMoeProcessorKwargs,
    Qwen3OmniMoeVideosKwargs,
    _get_feat_extract_output_lengths,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor as _Qwen3OmniMoeProcessor,
)
from transformers.tokenization_utils_base import TextInput
from transformers.video_utils import VideoInput, make_batched_videos


class Qwen3OmniMoeProcessor(_Qwen3OmniMoeProcessor):
    """
    VeOmni customized Qwen3OmniMoe processor.

    Modification:
        Use explicit `is not None` checks instead of truthy checks for audio/images/videos inputs.
        This prevents empty lists `[]` from being incorrectly treated as falsy values.
    """

    def __call__(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s).

        Modification from HuggingFace version:
            Use `if audio is not None` instead of `if audio` to properly handle empty lists.
        """
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3OmniMoeProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].get("fps", 1.0)

        # VeOmni modification: use truthy check
        if audio:
            output_kwargs["audio_kwargs"]["padding"] = True  # Setting to True to avoid default truncation
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        # VeOmni modification: use truthy check
        if images:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        # VeOmni modification: use truthy check
        if videos:
            videos = make_batched_videos(videos)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )


__all__ = [
    "Qwen3OmniMoeProcessor",
    "Qwen3OmniMoeProcessorKwargs",
    "Qwen3OmniMoeVideosKwargs",
    "Qwen3OmniMoeImagesKwargs",
]
