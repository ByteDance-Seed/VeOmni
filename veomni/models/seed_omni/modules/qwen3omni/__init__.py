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
"""Qwen3-Omni as SeedOmni modules.

An upstream ``Qwen3OmniMoeForConditionalGeneration`` checkpoint is a container
over ``thinker`` / ``talker`` / ``code2wav``. It splits into six modules::

    qwen3omni_audio         thinker.audio_tower   -- speech/sound -> embeds
    qwen3omni_vision        thinker.visual        -- image/video  -> embeds
    qwen3omni_text_encoder  thinker embed_tokens + lm_head + tokenizer
    qwen3omni_llm           thinker.model         -- AR backbone (no wte/head)
    qwen3omni_talker        talker                -- hidden states -> RVQ codes
    qwen3omni_codec         code2wav + kyutai/mimi -- waveform <-> RVQ codes

The codec is one module for both directions because Qwen3-Omni released only
half of it: ``code2wav`` decodes, but the matching audio tokenizer was never
published. The talker's codes turn out to live in Mimi's index space, so
``kyutai/mimi`` supplies the encoder and talker targets can be built from raw
audio. ``scripts/seed_omni/qwen3omni_audio_check.py`` measures how far that
sharing actually holds — audio-level yes, exact code-level no, and the control
that distinguishes the two matters. See ``codec/modeling.py``.
"""

from . import audio, codec, convert_model, hf_layout, llm, talker, text_encoder, vision  # noqa: F401
