# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
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
Processor class for Qwen3OmniMoe.
"""

import transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe as hf_qwen3_omni_moe
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor as _Qwen3OmniMoeProcessor,
)
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessorKwargs,
    _get_feat_extract_output_lengths,
)
from transformers.video_utils import make_batched_videos


# ================================================================
# Patch: Qwen3OmniMoeProcessor
# 1. Use truthy check `if audio:` instead of `if audio is not None:`
#    to properly handle empty lists from VeOmni data format
# ================================================================
class Qwen3OmniMoeProcessor(_Qwen3OmniMoeProcessor):
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        **kwargs,
    ) -> BatchFeature:
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

        # --- VeOmni Patch: use truthy check instead of `is not None` ---
        if audio:
            output_kwargs["audio_kwargs"]["padding"] = True
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        # --- VeOmni Patch: use truthy check instead of `is not None` ---
        if images:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        # --- VeOmni Patch: use truthy check instead of `is not None` ---
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


def apply_veomni_qwen3_omni_moe_patch():
    hf_qwen3_omni_moe.Qwen3OmniMoeProcessor = Qwen3OmniMoeProcessor


__all__ = ["Qwen3OmniMoeProcessor"]
