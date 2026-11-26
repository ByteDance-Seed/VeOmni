# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""DeepSeek-V4.1 token embedding + LM head (``deepseek_v41_text_encoder``).

The module is the generic :class:`TextEncoder` -- ``embed_tokens`` + an untied
``lm_head`` -- plus the V4.1 chat template and optional module-level freezing.

It also owns the shared embedding and LM head used by the DSpark draft stages
(``deepseek_v41_mtp``): ``mtp.*`` in the released checkpoint aliases the
backbone's ``embed`` / ``head`` rather than carrying its own.
"""

from ...text_encoder.modeling import TextEncoder
from .configuration import DeepseekV41TextEncoderConfig
from .processing import DeepseekV41TextEncoderPreprocessor


class DeepseekV41TextEncoder(TextEncoder):
    """V4.1 ``TextEncoder`` -- DeepSeek chat templating + tokenize + wte / lm_head.

    Image items pass through ``encode`` untouched: ``deepseek_v41_vision`` has
    already replaced their value with the span's ``(L, D)`` embeddings (delimiters
    included), and the backbone splices them by segment order. Only ``text`` rows
    are tokenized and embedded here.
    """

    config_class = DeepseekV41TextEncoderConfig
    preprocessor_class = DeepseekV41TextEncoderPreprocessor
    base_model_prefix = "deepseek_v41_text_encoder"

    def freeze_model(self) -> None:
        """Freeze ``embed_tokens`` and ``lm_head`` together when ``config.freeze`` is set."""
        if self.config.freeze:
            self.requires_grad_(False)


__all__ = ["DeepseekV41TextEncoder"]
