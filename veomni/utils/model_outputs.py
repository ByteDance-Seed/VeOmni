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

"""Model output dataclass for the per-token log-probs and top-k distillation paths.

A patched ``*ForCausalLM.forward`` returns this dataclass when called
with ``return_log_probs=True``: ``log_probs`` carries per-token actual
log-probabilities (non-positive), ``entropy`` carries per-token softmax
entropy (non-negative); ``logits`` and ``loss`` are ``None``. When the
caller additionally passes ``teacher_topk_ids`` and
``teacher_topk_log_probs``, the wrapper short-circuits to the top-k
forward-KL distillation kernel and the three trailing fields
(``distillation_losses``, ``student_mass``, ``teacher_mass``) carry
verl's per-token distillation tensors. Imports are kept light (no
``veomni.data`` dependency) so external integrators (verl) can pull the
dataclass without paying the data-pipeline import cost.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerCausalLMOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast


@dataclass
class CausalLMOutputWithLogProbs(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` extended with per-token log-probs / distillation fields.

    All five tensors share the input ``labels`` shape (``[B, L]`` or
    packed ``[L]``) and are zero at IGNORE_INDEX positions and the
    trailing pad slot.

    - ``log_probs``: non-positive — actual log-probabilities ``log p(y_t)``,
      matches HF / verl conventions.
    - ``entropy``: non-negative — softmax entropy
      ``H[p] = -Σ_v p_v log p_v``, matches verl's
      ``CausalLMOutputForPPO.entropy`` so the dataclass drops directly
      into verl's ``prepare_model_outputs`` consumer.
    - ``distillation_losses``: non-negative — top-k forward KL
      ``Σ_k exp(log p_t,k) (log p_t,k - log q_s,k)``, matching verl's
      ``compute_forward_kl_topk`` output key. Carries gradient back to
      the lm_head + hidden_states.
    - ``student_mass`` / ``teacher_mass``: non-negative metric tensors,
      ``Σ_k exp(log q_s,k)`` / ``Σ_k exp(log p_t,k)``. Detached — verl
      uses them for clamp monitoring and reporting, not for backprop.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class MoeCausalLMOutputWithLogProbs(MoeCausalLMOutputWithPast):
    """``MoeCausalLMOutputWithPast`` extended with per-token log-probs / distillation fields.

    All five tensors share the input ``labels`` shape (``[B, L]`` or
    packed ``[L]``) and are zero at IGNORE_INDEX positions and the
    trailing pad slot.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``, matching HF
            and verl conventions.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``, matching
            verl's ``CausalLMOutputForPPO.entropy`` so the dataclass drops
            directly into verl's ``prepare_model_outputs`` consumer.
        distillation_losses (`torch.FloatTensor`, *optional*):
            Per-token top-k forward KL, matches verl's
            ``compute_forward_kl_topk`` output. Backprops through the
            lm_head + hidden_states.
        student_mass (`torch.FloatTensor`, *optional*):
            Detached metric: student's probability mass on the teacher's
            top-k support.
        teacher_mass (`torch.FloatTensor`, *optional*):
            Detached metric: teacher's probability mass on its own top-k
            support (clamp-aware when ``log_prob_min_clamp`` is set).
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


# ──────────────────────────────────────────────────────────────────────────────
# Model-specific subclasses for multimodal/omni outputs.
#
# These mirror the HF base classes (preserving ``rope_deltas`` and other
# model-specific fields) and add ``log_probs`` / ``entropy`` as constructor
# fields. They live here (rather than inline in each patchgen config) so the
# GPU and NPU generated modeling files can share one definition.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Qwen2VLCausalLMOutputWithLogProbs(Qwen2VLCausalLMOutputWithPast):
    """``Qwen2VLCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class Qwen2_5_VLCausalLMOutputWithLogProbs(Qwen2_5_VLCausalLMOutputWithPast):
    """``Qwen2_5_VLCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class Qwen3VLCausalLMOutputWithLogProbs(Qwen3VLCausalLMOutputWithPast):
    """``Qwen3VLCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class Qwen3VLMoeCausalLMOutputWithLogProbs(Qwen3VLMoeCausalLMOutputWithPast):
    """``Qwen3VLMoeCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithLogProbs(Qwen2_5OmniThinkerCausalLMOutputWithPast):
    """``Qwen2_5OmniThinkerCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None


@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithLogProbs(Qwen3OmniMoeThinkerCausalLMOutputWithPast):
    """``Qwen3OmniMoeThinkerCausalLMOutputWithPast`` extended with per-token ``log_probs`` / ``entropy`` fields.

    Args:
        log_probs (`torch.FloatTensor`, *optional*):
            Non-positive actual log-probabilities ``log p(y_t)``.
        entropy (`torch.FloatTensor`, *optional*):
            Non-negative softmax entropy ``H[p] = -Σ_v p_v log p_v``.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None
