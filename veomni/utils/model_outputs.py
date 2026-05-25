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

"""Model output dataclasses for the per-token fused-linear loss path.

A patched ``*ForCausalLM.forward`` returns one of the ``*WithLogProbs``
dataclasses below. When called with ``return_log_probs=True``, the
``fused_linear_aux`` field carries a ``FusedLinearAuxOutput`` payload
holding the per-token tensors verl's distillation and PPO consumers
read; ``logits`` and ``loss`` are then ``None``. On the plain loss
path ``fused_linear_aux`` is ``None`` and ``logits`` / ``loss`` are
populated as usual.

Two-level shape (nested payload + thin mixin) keeps the per-model
subclass declarations to a single shared field — adding a new
per-token metric only edits ``FusedLinearAuxOutput`` (one place),
not every ``*WithLogProbs`` subclass + every patchgen ``forward``.
Imports are kept light (no ``veomni.data`` dependency) so external
integrators (verl) can pull the dataclasses without paying the
data-pipeline import cost.
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
class FusedLinearAuxOutput:
    """Per-token tensors produced by the fused-linear loss path.

    All five tensors share the input ``labels`` shape (``[B, L]`` or
    packed ``[L]``) and are zero at IGNORE_INDEX positions and the
    trailing pad slot.

    - ``log_probs``: non-positive — actual log-probabilities
      ``log p(y_t)``, matches HF / verl conventions.
    - ``entropy``: non-negative — softmax entropy
      ``H[p] = -Σ_v p_v log p_v``, matches verl's
      ``CausalLMOutputForPPO.entropy`` so the payload drops directly
      into verl's ``prepare_model_outputs`` consumer.
    - ``distillation_losses``: non-negative (in the full-support
      limit) — top-k forward KL
      ``Σ_k exp(log p_t,k) (log p_t,k - log q_s,k)``, matching verl's
      ``compute_forward_kl_topk`` output key. Carries gradient back
      to the lm_head + hidden_states.
    - ``student_mass`` / ``teacher_mass``: non-negative metric
      tensors, ``Σ_k exp(log q_s,k)`` / ``Σ_k exp(log p_t,k)``.
      Detached — verl uses them for clamp monitoring and reporting,
      not for backprop.
    """

    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    distillation_losses: Optional[torch.Tensor] = None
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None

    @classmethod
    def from_loss_slots(
        cls,
        log_probs: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        distillation_losses: Optional[torch.Tensor] = None,
        student_mass: Optional[torch.Tensor] = None,
        teacher_mass: Optional[torch.Tensor] = None,
    ) -> Optional["FusedLinearAuxOutput"]:
        """Construct from the loss-wrapper's trailing 5 slots, or return
        ``None`` if all slots are ``None`` (the plain loss path).

        Keeps the patchgen ``forward`` template a one-liner regardless
        of which branch ran inside ``self.loss_function``.
        """
        if (
            log_probs is None
            and entropy is None
            and distillation_losses is None
            and student_mass is None
            and teacher_mass is None
        ):
            return None
        return cls(
            log_probs=log_probs,
            entropy=entropy,
            distillation_losses=distillation_losses,
            student_mass=student_mass,
            teacher_mass=teacher_mass,
        )


@dataclass
class FusedLinearAuxOutputMixin:
    """Single ``fused_linear_aux`` field added to every ``*WithLogProbs``
    dataclass. Inherited alongside the HF base class so per-model
    subclasses don't repeat the field.

    Also exposes ``log_probs`` and ``entropy`` as read-only properties
    that proxy to ``fused_linear_aux``. Restores the pre-#780 attribute
    surface so external consumers (notably verl's
    ``prepare_model_outputs`` at
    ``verl/workers/engine/{fsdp,automodel}/transformer_impl.py``) can keep
    reading ``output.log_probs`` / ``output.entropy`` directly without
    knowing about the nested ``FusedLinearAuxOutput`` payload.
    """

    fused_linear_aux: Optional[FusedLinearAuxOutput] = None

    @property
    def log_probs(self) -> Optional[torch.Tensor]:
        return self.fused_linear_aux.log_probs if self.fused_linear_aux is not None else None

    @property
    def entropy(self) -> Optional[torch.Tensor]:
        return self.fused_linear_aux.entropy if self.fused_linear_aux is not None else None


_FUSED_LINEAR_AUX_ARGS_DOC = """
    Args:
        fused_linear_aux (`FusedLinearAuxOutput`, *optional*):
            Per-token tensors produced by the fused-linear loss path
            (``log_probs``, ``entropy``, ``distillation_losses``,
            ``student_mass``, ``teacher_mass``). ``None`` on the plain
            loss path; populated when ``return_log_probs=True``.
    """


@dataclass
class CausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, CausalLMOutputWithPast):
    __doc__ = "``CausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


@dataclass
class MoeCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, MoeCausalLMOutputWithPast):
    __doc__ = "``MoeCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


# ──────────────────────────────────────────────────────────────────────────────
# Model-specific subclasses for multimodal/omni outputs.
#
# These mirror the HF base classes (preserving ``rope_deltas`` and other
# model-specific fields) and pick up ``fused_linear_aux`` via the mixin. They
# live here (rather than inline in each patchgen config) so the GPU and NPU
# generated modeling files can share one definition.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Qwen2VLCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen2VLCausalLMOutputWithPast):
    __doc__ = "``Qwen2VLCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


@dataclass
class Qwen2_5_VLCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen2_5_VLCausalLMOutputWithPast):
    __doc__ = "``Qwen2_5_VLCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


@dataclass
class Qwen3VLCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen3VLCausalLMOutputWithPast):
    __doc__ = "``Qwen3VLCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


@dataclass
class Qwen3VLMoeCausalLMOutputWithLogProbs(FusedLinearAuxOutputMixin, Qwen3VLMoeCausalLMOutputWithPast):
    __doc__ = "``Qwen3VLMoeCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithLogProbs(
    FusedLinearAuxOutputMixin, Qwen2_5OmniThinkerCausalLMOutputWithPast
):
    __doc__ = (
        "``Qwen2_5OmniThinkerCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC
    )


@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithLogProbs(
    FusedLinearAuxOutputMixin, Qwen3OmniMoeThinkerCausalLMOutputWithPast
):
    __doc__ = (
        "``Qwen3OmniMoeThinkerCausalLMOutputWithPast`` + ``fused_linear_aux`` payload." + _FUSED_LINEAR_AUX_ARGS_DOC
    )
