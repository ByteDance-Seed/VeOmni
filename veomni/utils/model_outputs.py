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

"""Model output dataclass for the per-token log-probs path.

A patched ``*ForCausalLM.forward`` returns this dataclass when called
with ``return_log_probs=True``: ``log_probs`` carries per-token actual
log-probabilities (non-positive), ``logits`` and ``loss`` are
``None``. Imports are kept light (no ``veomni.data`` dependency) so
external integrators (verl) can pull the dataclass without paying the
data-pipeline import cost.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputWithLogProbs(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` extended with a per-token ``log_probs`` field.

    ``log_probs`` shape matches the input ``labels`` (``[B, L]`` or
    packed ``[L]``). Sign: non-positive — actual log-probabilities,
    matches HF / verl conventions. Zero at IGNORE_INDEX positions and
    the trailing pad slot.
    """

    log_probs: Optional[torch.Tensor] = None
