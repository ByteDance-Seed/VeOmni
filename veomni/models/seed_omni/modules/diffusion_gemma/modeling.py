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

"""DiffusionGemma as a single SeedOmni module.

One node, like the LLaDA families, but the node is two passes internally: the
encoder prefills the prompt and exports a KV cache, and the decoder denoises the
drawn canvas against it. Nothing crosses a module boundary -- the cache stays
inside this call -- so there is still nothing to hand to a sibling.

Only ``_loss`` leaves this module. The vendored forward also returns
``diffusion_loss`` and ``encoder_ar_loss`` as the two ``Eq. 13`` components, but
``OmniModel.forward`` collects nothing but the ``_loss`` key, so a component cannot
reach the trainer's log through this graph. Reporting them would need a
component-loss channel in the graph runtime; until that exists, the split is
visible only through the non-Omni trainer. The summed ``_loss`` is unaffected --
the vendored forward already added the two terms together.
"""

from __future__ import annotations

from typing import Any

from ....transformers.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion
from ..module_modeling_base import PretrainedOmniModule
from .configuration import DiffusionGemmaOmniConfig
from .processing import DiffusionGemmaOmniPreprocessor


class DiffusionGemmaOmni(DiffusionGemmaForBlockDiffusion, PretrainedOmniModule):
    """The vendored block-diffusion model, registered as an OmniModule.

    ``DiffusionGemmaForBlockDiffusion`` comes first so its ``forward`` (encoder
    prefill, decoder denoising, ``Eq. 13`` loss assembly) is the one the graph
    invokes; ``PretrainedOmniModule`` contributes the asset binding and the no-op
    inference defaults.

    ``preprocessor_class`` is what ``OmniProcessor.from_config`` reads to build the
    CPU worker; without it the sample reaches the graph unprocessed and ``batch_pre``
    finds no tensors to stack.
    """

    config_class = DiffusionGemmaOmniConfig
    preprocessor_class = DiffusionGemmaOmniPreprocessor
    base_model_prefix = "diffusion_gemma_omni"

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Run the encoder/decoder pair and surface ``_loss`` plus its two components.

        The graph passes the collated micro-batch as keyword arguments, so this
        forwards every canvas input straight to the vendored implementation. The
        component losses are passed through for logging only -- ``loss`` already
        contains them, so reporting them must not add them a second time.
        """
        kwargs.pop("conversation_list", None)
        # `num_supervised_examples` is the collator's own bookkeeping for the
        # trainer's per-sample divisor; the module counts samples itself from the
        # carrier, so forwarding it would let two counters disagree.
        kwargs.pop("num_supervised_examples", None)
        outputs = super().forward(**kwargs)
        result: dict[str, Any] = {}
        if outputs.loss is not None:
            result["_loss"] = outputs.loss
        aux_loss = getattr(outputs, "aux_loss", None)
        if aux_loss is not None:
            result["aux_loss"] = aux_loss
        return result


__all__ = ["DiffusionGemmaOmni"]
