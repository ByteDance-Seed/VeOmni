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

"""LLaDA-MoE as a single SeedOmni module.

Same shape as the dense :mod:`..llada` module: the mask predictor is
supervised in place over one packed row, so the graph is one node deep. The only
additions are the router load-balancing term and the Expert Parallel plan, both
of which live in ``accelerated/``.

``_loss`` stays the *local weighted sum* ``sum_i w_i * CE_i`` and ``aux_loss``
travels beside it, unweighted and unfolded: the objective divides the CE term by
the step's global sample count, and folding a per-token mean into that would
rescale it with the batch composition. ``LladaMoeTrainingMixin.normalize_loss``
adds ``router_aux_loss_coef * aux_loss`` after the division, matching
``DllmTrainer.postforward``.
"""

from __future__ import annotations

from typing import Any

import torch

from ....transformers.llada_moe.modeling_llada_moe import LLaDAMoEForCausalLM
from ..llada.modeling import LladaOmni
from ..module_modeling_base import PretrainedOmniModule
from .configuration import LladaMoeOmniConfig
from .processing import LladaMoeOmniPreprocessor


class LladaMoeOmni(LLaDAMoEForCausalLM, PretrainedOmniModule):
    """The vendored LLaDA-MoE model, registered as an OmniModule.

    ``LLaDAMoEForCausalLM`` comes first so its ``forward`` (bidirectional mask plus
    ``dllm_lm_head_loss``) is the one the graph invokes;
    ``PretrainedOmniModule`` contributes the asset binding and the no-op inference
    defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build the
    CPU worker; without it the sample reaches the graph unprocessed and
    ``batch_pre`` finds no tensors to stack.

    The denoising loop is **not** restated here. LLaDA-MoE is the same mask
    predictor over the same row layout -- the router changes how a token is
    computed, not which positions are masked or in what order they are committed
    -- so the generation methods are borrowed from the dense module rather than
    copied. A second copy is how the two would drift apart on the one thing that
    matters: which positions count as decoded.
    """

    config_class = LladaMoeOmniConfig
    preprocessor_class = LladaMoeOmniPreprocessor
    base_model_prefix = "llada_moe_omni"
    _no_split_modules = ["LLaDAMoEDecoderLayer"]

    # Borrowed from the dense module: the loop reads only `config.mask_token_id`,
    # the tokenizer and its EOS, all of which this module also holds.
    SIGNAL_TEXT_DONE = LladaOmni.SIGNAL_TEXT_DONE
    tokenizer = LladaOmni.tokenizer
    _eos_token_id = LladaOmni._eos_token_id
    _reset_diffusion_state = LladaOmni._reset_diffusion_state
    _make_num_transfer_schedule = staticmethod(LladaOmni._make_num_transfer_schedule)
    _prompt_ids = LladaOmni._prompt_ids
    _diffusion_init = LladaOmni._diffusion_init
    _diffusion_step = LladaOmni._diffusion_step
    _finish_diffusion = LladaOmni._finish_diffusion
    generate = LladaOmni.generate
    finalize = LladaOmni.finalize

    def __init__(self, config: LladaMoeOmniConfig):
        super().__init__(config)
        self._reset_diffusion_state()

    def reset_local_inference_state(self) -> None:
        """Clear the denoising row between requests.

        Stated here rather than borrowed: the dense module's version calls
        ``super().reset_local_inference_state()``, and a zero-argument ``super()``
        is bound to the class it was written in -- borrowing it makes the call
        fail with "obj must be an instance or subtype of type", because ``self``
        is this class and not that one.
        """
        super().reset_local_inference_state()
        self._reset_diffusion_state()

    def _decode_forward(self) -> dict[str, Any]:
        """Run the mask predictor over the whole row and return its logits.

        One line different from the dense module -- the LM head is applied here --
        and kept separate rather than borrowed so the backbone attribute this
        class uses is stated in this class.
        """
        x = self._diff_x
        if x is None:
            raise RuntimeError("LladaMoeOmni: decode forward without an initialised row.")
        outputs = self.model(
            input_ids=x.unsqueeze(0),
            attention_mask=torch.ones_like(x, dtype=torch.long).unsqueeze(0),
            use_cache=False,
        )
        return {"llada_logits": self.lm_head(outputs.last_hidden_state)}

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # LLaDA-MoE owns attention and validates its backend: the bidirectional
        # mask has to be representable by the chosen kernel.
        return super()._check_and_adjust_attn_implementation(attn_implementation, **kwargs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the mask predictor and surface ``_loss`` plus ``aux_loss``.

        The graph passes the collated micro-batch as keyword arguments, so this
        forwards ``input_ids`` / ``attention_mask`` / ``labels`` / ``loss_weight``
        straight to the vendored implementation and moves its scalar loss to the
        key the executor reads back. ``aux_loss`` is passed through untouched: it
        is folded in after the sample-count division, not here.
        """
        kwargs.pop("conversation_list", None)
        outputs = super().forward(use_cache=False, **kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            result["aux_loss"] = aux_loss
        return result


__all__ = ["LladaMoeOmni"]
