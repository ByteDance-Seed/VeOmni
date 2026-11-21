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

"""LLaDA as a single SeedOmni module.

LLaDA is a mask predictor, not a decoder: it is supervised in place, over the
masked positions of the very sequence it is given. That makes it a
one-node graph -- there is no second module to hand anything to, and the CPU
preprocessor has already drawn the corruption before the batch reaches the GPU.

The forward returns ``_loss`` as the *local weighted sum* ``sum_i w_i * CE_i``
with ``w_i = 1/(t_s * N_s)`` baked in by the collator, exactly as
``LLaDAForCausalLM`` produces it. Dividing by the step's global sample count is
the trainer's job and stays where it is for the non-Omni path
(``dllm_mean_global_loss``); the Omni wrapper does the same division in
``LladaOmniTrainingMixin.post_forward`` using the count
``on_micro_batches_ready`` recorded for this step, and declares
``loss_is_step_mean`` so the trainer does not divide the backward again.
"""

from __future__ import annotations

from typing import Any

from ....transformers.llada.modeling_llada import LLaDAForCausalLM
from ..module_modeling_base import PretrainedOmniModule
from .configuration import LladaOmniConfig
from .processing import LladaOmniPreprocessor


class LladaOmni(LLaDAForCausalLM, PretrainedOmniModule):
    """The vendored LLaDA model, registered as an OmniModule.

    ``LLaDAForCausalLM`` comes first so its ``forward`` (which knows how to build
    the bidirectional mask and call ``dllm_lm_head_loss``) is the one the graph
    invokes; ``PretrainedOmniModule`` contributes the asset binding and the
    no-op inference defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build
    the CPU worker; without it the sample would reach the graph unprocessed and
    ``batch_pre`` would find no tensors to stack.
    """

    config_class = LladaOmniConfig
    preprocessor_class = LladaOmniPreprocessor
    base_model_prefix = "llada_omni"
    _no_split_modules = ["LLaDADecoderLayer"]

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # LLaDA owns attention and really does validate its backend: the
        # bidirectional mask has to be representable by the chosen kernel.
        return super()._check_and_adjust_attn_implementation(attn_implementation, **kwargs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the mask predictor over the carrier's tensors and surface ``_loss``.

        The graph passes the collated micro-batch as keyword arguments, so this
        forwards ``input_ids`` / ``attention_mask`` / ``labels`` / ``loss_weight``
        straight to the vendored implementation and moves its scalar loss to the
        key the executor reads back.
        """
        # `logits_to_keep` and friends are not part of the training contract; a
        # training step supervises the masked positions and never materialises
        # full logits (the LM head runs on the selected rows inside the loss).
        kwargs.pop("conversation_list", None)
        outputs = super().forward(use_cache=False, **kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            result["aux_loss"] = aux_loss
        return result


__all__ = ["LladaOmni"]
