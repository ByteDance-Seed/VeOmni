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

import inspect
from typing import List, Optional

import numpy as np
from transformers import BatchFeature, WhisperFeatureExtractor


try:
    from transformers.audio_utils import AudioInput
except Exception:
    from transformers.tokenization_utils_base import AudioInput

from ..base import BaseEncoderProcessorMixin


class Qwen25OmniAudioModelProcessor(BaseEncoderProcessorMixin, WhisperFeatureExtractor):
    valid_kwargs = BaseEncoderProcessorMixin.valid_kwargs + list(
        inspect.signature(WhisperFeatureExtractor.__init__).parameters.keys()
    )

    def __init__(
        self,
        token_num: int = None,
        token_size: List = None,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
        **kwargs,
    ) -> None:
        BaseEncoderProcessorMixin.__init__(self, token_num=token_num, token_size=token_size, **kwargs)
        WhisperFeatureExtractor.__init__(
            self,
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            dither=dither,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

    @property
    def model_input_names(self):
        return WhisperFeatureExtractor.model_input_names

    def process(
        self,
        audios: Optional[AudioInput] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        audios = [audio if audio is not None else np.zeros((0,)) for audio in audios]
        output = self.__call__(
            audios,
            return_tensors=return_tensors,
            return_attention_mask=True,
            padding="max_length",
            sampling_rate=self.sampling_rate,
        )
        features = output["input_features"]
        mask = output["attention_mask"]
        feature_lengths = mask.sum(-1)
        input_lengths = (feature_lengths - 1) // 2 + 1
        num_tokens = (input_lengths - 2) // 2 + 1
        valid_mask = num_tokens > 0
        features = features[valid_mask].permute(0, 2, 1)[mask[valid_mask].bool()]  # (len, dim)
        return BatchFeature(
            data={"features": features, "num_tokens": num_tokens, "feature_lengths": feature_lengths},
            tensor_type=return_tensors,
        )
