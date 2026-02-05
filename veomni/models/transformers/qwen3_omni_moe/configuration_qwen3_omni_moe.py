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
VeOmni customized Qwen3OmniMoe configuration.

This module extends the HuggingFace Qwen3OmniMoe configuration with VeOmni-specific modifications.
"""

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig as _Qwen3OmniMoeConfig,
)


class Qwen3OmniMoeConfig(_Qwen3OmniMoeConfig):
    """
    VeOmni customized Qwen3OmniMoe configuration.

    Modification:
        Qwen3OmniMoe didn't set tie_word_embeddings, so it is set to True by default.
        However, Qwen3OmniMoe model didn't set get_output_embeddings, so the `embed_tokens` can't tie with `lm_head`.
        Logically, `tie_word_embeddings=False`.
    """

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        enable_audio_output=True,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        system_token_id=8948,
        user_token_id=872,
        assistant_token_id=77091,
        **kwargs,
    ):
        # Force tie_word_embeddings=False since Qwen3OmniMoe doesn't implement get_output_embeddings
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            thinker_config=thinker_config,
            talker_config=talker_config,
            code2wav_config=code2wav_config,
            enable_audio_output=enable_audio_output,
            im_start_token_id=im_start_token_id,
            im_end_token_id=im_end_token_id,
            tts_pad_token_id=tts_pad_token_id,
            tts_bos_token_id=tts_bos_token_id,
            tts_eos_token_id=tts_eos_token_id,
            system_token_id=system_token_id,
            user_token_id=user_token_id,
            assistant_token_id=assistant_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )


__all__ = [
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeVisionEncoderConfig",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeCode2WavConfig",
]
