# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os


try:
    from hdfs_io import exists  # for internal use only
except ImportError:
    from ...utils.hdfs_io import exists
from transformers import AutoProcessor, BatchFeature, ProcessorMixin

from ...utils import logging


logger = logging.get_logger(__name__)


class SeedOmniProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    optional_attributes = (
        ["input_image_processor", "input_video_processor", "input_audio_processor"]
        + ["output_image_processor", "output_video_processor", "output_audio_processor"]
        + ["target_image_processor", "target_video_processor", "target_audio_processor"]
        + ["chat_template", "audio_tokenizer"]
    )
    input_image_processor_class = "AutoProcessor"
    input_audio_processor_class = "AutoProcessor"
    input_video_processor_class = "AutoProcessor"
    output_image_processor_class = "AutoProcessor"
    output_audio_processor_class = "AutoProcessor"
    output_video_processor_class = "AutoProcessor"
    target_image_processor_class = "AutoProcessor"
    target_audio_processor_class = "AutoProcessor"
    target_video_processor_class = "AutoProcessor"
    tokenizer_class = "AutoTokenizer"
    processor_prefixes = [
        "input_image",
        "output_image",
        "target_image",
        "input_video",
        "output_video",
        "target_video",
        "input_audio",
        "output_audio",
        "target_audio",
    ]

    def __init__(self, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        input_images=None,
        output_images=None,
        input_videos=None,
        output_videos=None,
        input_audios=None,
        output_audios=None,
        **kwargs,
    ) -> BatchFeature:
        """For input & output processor:
        features, grid_thw, num_tokens
        For target processor:
        features, kwargs
        """
        inputs = {}
        if input_images:
            image_inputs: BatchFeature = self.input_image_processor.process(images=input_images, **kwargs)
            for key, value in image_inputs.items():
                inputs[f"image_input_{key}"] = value

        if output_images:
            image_inputs: BatchFeature = self.output_image_processor.process(images=output_images, **kwargs)
            for key, value in image_inputs.items():
                inputs[f"image_output_{key}"] = value

            if getattr(self, "target_image_processor", None) is not None:  # target for diffusion models
                image_inputs: BatchFeature = self.target_image_processor.process(images=output_images, **kwargs)
                for key, value in image_inputs.items():
                    inputs[f"image_target_{key}"] = value

        if input_videos:
            if getattr(self, "input_video_processor", None) is not None:
                video_inputs: BatchFeature = self.input_video_processor.process(videos=input_videos, **kwargs)
            else:
                logger.warning_once("The input_video_processor is not set, use input_image_processor.")
                video_inputs: BatchFeature = self.input_image_processor.process(videos=input_videos, **kwargs)

            for key, value in video_inputs.items():
                inputs[f"video_input_{key}"] = value

        if input_audios:
            audio_inputs: BatchFeature = self.input_audio_processor.process(audios=input_audios, **kwargs)
            for key, value in audio_inputs.items():
                inputs[f"audio_input_{key}"] = value

        if output_videos or output_audios:
            raise NotImplementedError

        return BatchFeature(data=inputs)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        for prefix in self.processor_prefixes:
            processor = getattr(self, f"{prefix}_processor", None)
            if processor is not None:
                save_path = os.path.join(save_directory, f"{prefix}_processor")
                processor.save_pretrained(save_path)

        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # if return_unused_kwargs a tuple is returned where the second element is "unused_kwargs"
        if isinstance(processor, tuple):
            processor = processor[0]

        for prefix in cls.processor_prefixes:
            if exists(pretrained_model_name_or_path + f"/{prefix}_processor"):
                sub_processor = AutoProcessor.from_pretrained(
                    os.path.join(pretrained_model_name_or_path, f"{prefix}_processor")
                )
                setattr(processor, f"{prefix}_processor", sub_processor)
            else:
                setattr(processor, f"{prefix}_processor", None)

        return processor
