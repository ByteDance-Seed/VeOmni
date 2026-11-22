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

"""LLaDA-V as a single SeedOmni module.

Same objective and same one-node graph as the dense :mod:`..llada` module: the
mask predictor supervises itself in place over one packed row. What this family
adds is the SigLIP tower and the projector, which run inside the same node -- the
image rows are scattered into the text embeddings before the backbone, so there is
still nothing to hand to a sibling module.

``_init_weights`` is inherited from ``LLaDAVPreTrainedModel`` and must stay that
way: ``to_empty()`` on the meta random-init path discards the constructor's zeros
for ``image_newline``, and nothing in the checkpoint restores it, so the hook
re-applies them after materialisation.
"""

from __future__ import annotations

from typing import Any

from ....transformers.llada_v.modeling_llada_v import LLaDAVForConditionalGeneration
from ..module_modeling_base import PretrainedOmniModule
from .configuration import LladaVOmniConfig
from .processing import LladaVOmniPreprocessor


class LladaVOmni(LLaDAVForConditionalGeneration, PretrainedOmniModule):
    """The vendored LLaDA-V model, registered as an OmniModule.

    ``LLaDAVForConditionalGeneration`` comes first so its ``forward`` (vision
    scatter, bidirectional mask, ``dllm_lm_head_loss``) is the one the graph invokes;
    ``PretrainedOmniModule`` contributes the asset binding and the no-op inference
    defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build the
    CPU worker; without it the sample reaches the graph unprocessed and
    ``batch_pre`` finds no tensors to stack.
    """

    config_class = LladaVOmniConfig
    preprocessor_class = LladaVOmniPreprocessor
    base_model_prefix = "llada_v_omni"
    _no_split_modules = ["LLaDADecoderLayer", "SiglipEncoderLayer"]

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        # LLaDA-V owns attention on the backbone and validates its backend. The
        # vision tower is pinned to sdpa inside the vendored model, so it cannot
        # inherit a flash kernel that would all-to-all the patch sequence.
        return super()._check_and_adjust_attn_implementation(attn_implementation, **kwargs)

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the mask predictor and surface ``_loss``.

        The graph passes the collated micro-batch as keyword arguments. ``pixel_values``
        and ``image_mask`` travel with it when the batch has an image; without them the
        vendored forward still runs the tower once (a dummy image) whenever the world
        size is above one, so the FSDP all-gathers of ranks that do have images stay
        matched.
        """
        kwargs.pop("conversation_list", None)
        outputs = super().forward(use_cache=False, **kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        return result


__all__ = ["LladaVOmni"]
