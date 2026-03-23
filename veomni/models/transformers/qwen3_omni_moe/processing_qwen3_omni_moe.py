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

import re

import numpy as np
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
# 2. Accept veomni multimodal data format: `audios``
# ================================================================
class Qwen3OmniMoeProcessor(_Qwen3OmniMoeProcessor):
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        # --- Patch.2 ---
        audios=None,
        # --- Patch.2 ---
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
        fps = output_kwargs["videos_kwargs"].get("fps", 1.0)

        # --- Patch.2 ---
        _ = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        # --- Patch.2 ---

        # Modification: use truthy check instead of `is not None`
        if audios:
            # --- Patch.2 ---
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audios = [audio if audio is not None else np.zeros((0,)) for audio in audios]
            # --- Patch.2 ---

            audio_inputs = self.feature_extractor(audios, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        # Modification: use truthy check instead of `is not None`
        if images:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        # Modification: use truthy check instead of `is not None`
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
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        # --- Patch.2 ---
        # use_audio_in_video,
        # --- Patch.2 ---
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    # --- Patch.2 ---
                    audio_length = next(audio_lengths)
                    use_audio_in_video = audio_length != 0
                    # --- Patch.2 ---

                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token, "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        # --- Patch.2 ---
                        audio_token_indices = np.arange(audio_length)
                        # --- Patch.2 ---
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        video_data_index, audio_data_index = 0, 0
                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        while video_data_index < len(video_token_indices) and audio_data_index < len(
                            audio_token_indices
                        ):
                            if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                                placeholder_string += "<|video_placeholder|>"
                                video_data_index += 1
                            else:
                                placeholder_string += "<|audio_placeholder|>"
                                audio_data_index += 1
                        if video_data_index < len(video_token_indices):
                            placeholder_string += "<|video_placeholder|>" * (
                                len(video_token_indices) - video_data_index
                            )
                        if audio_data_index < len(audio_token_indices):
                            placeholder_string += "<|audio_placeholder|>" * (
                                len(audio_token_indices) - audio_data_index
                            )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text


def apply_veomni_qwen3_omni_moe_patch():
    hf_qwen3_omni_moe.Qwen3OmniMoeProcessor = Qwen3OmniMoeProcessor


__all__ = ["Qwen3OmniMoeProcessor"]
