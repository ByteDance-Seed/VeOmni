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
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
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
    __doc__ = (
        "``MoeCausalLMOutputWithPast`` + ``fused_linear_aux`` payload, and scalar "
        "auxiliary metrics already folded into ``loss``."
        + _FUSED_LINEAR_AUX_ARGS_DOC
        + """
        aux_metrics (`dict[str, torch.Tensor]`, *optional*):
            Detached 0-d tensors a forward wants reported next to the loss --
            DeepSeek-V4's ``indexer_kl`` is the first. Whatever the forward chose
            to add to ``loss`` it has already added; the trainer reports these and
            must not sum them into the backward scalar a second time.
    """
    )

    aux_metrics: Optional[dict[str, torch.Tensor]] = None


@dataclass
class MoeModelOutputWithIndexerKL(MoeModelOutputWithPast):
    """``MoeModelOutputWithPast`` + the auxiliary KL a model body accumulated.

    Declared fields rather than attributes assigned onto the output object, which
    is what the DeepSeek-V4 indexer loss originally called for.
    ``ModelOutput.__setattr__`` writes into the underlying dict only for keys that
    already exist, so ``outputs.indexer_kl_total = kl`` on the base class creates a
    plain instance attribute: absent from ``keys()``, from ``to_tuple()`` and from
    pytree flattening, and dropped by any round-trip. The immediate read in
    ``ForCausalLM.forward`` would still work and the loss would look right, while
    FSDP2's pre-backward unshard hook -- which walks that same flattened output to
    find the tensors a backward will need -- would never see it.

    Every field below is ``None`` with the loss disabled, and ``ModelOutput.keys()``
    skips ``None``, so a flag-off output is indistinguishable from the base class's.

    The entries below are in HuggingFace's ``name (type):`` form because
    ``@auto_docstring`` parses this class -- it is the return annotation of a
    decorated ``forward`` -- and raises ``No `Args` or `Parameters` section is
    found`` on a docstring whose parameters it cannot match.

    Args:
        indexer_kl_total (`torch.Tensor`, *optional*):
            0-d, summed over the CSA layers and over this rank's query rows. Not
            yet divided by the token count: that division and the cross-rank
            reduction belong together, in ``ForCausalLM.forward``.
        indexer_uniform_total (`torch.Tensor`, *optional*):
            0-d, ``log(n_candidates) - H(target)`` accumulated over exactly the rows
            and layers ``indexer_kl_total`` covers: the KL a student would pay
            knowing the candidate set and nothing else. Detached -- it is the scale
            the KL is reported against and must never reach the objective. Summed
            here rather than divided per row so the reported quantity can be a ratio
            of means.
        indexer_query_tokens (`int`, *optional*):
            Number of local query rows the sums above cover.
        indexer_kl_layers (`int`, *optional*):
            Number of CSA layers that contributed to the sums. The loss keeps the
            sum; the reported metric divides by this, so runs with different CSA
            layer counts are comparable.
    """

    indexer_kl_total: Optional[torch.Tensor] = None
    indexer_uniform_total: Optional[torch.Tensor] = None
    indexer_query_tokens: Optional[int] = None
    indexer_kl_layers: Optional[int] = None


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
