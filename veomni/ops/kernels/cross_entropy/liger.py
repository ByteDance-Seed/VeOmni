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

from typing import Optional

import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


liger_kernel_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")


def fused_liger_kernel_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)
    if hidden_states is None or weights is None:
        # Fused linear + CE avoids materializing the full logits tensor by
        # doing the projection inside the Liger kernel — which means it needs
        # the pre-projection hidden states *and* the projection weights. If
        # either is missing, the model's forward was not patched to pass them
        # through to ``self.loss_function`` / the OpSlot. Fail loudly so the
        # user either patches the model or consciously switches to eager.
        raise RuntimeError(
            "fused_liger_kernel_cross_entropy requires `hidden_states` and `weights` "
            f"(got hidden_states={'set' if hidden_states is not None else 'None'}, "
            f"weights={'set' if weights is not None else 'None'}).\n"
            "Fix: patch the model's `ForCausalLM.forward` / "
            "`ForSequenceClassification.forward` to pass "
            "`hidden_states=hidden_states, weights=self.lm_head.weight` into "
            "`self.loss_function(...)` — see "
            "`veomni/models/transformers/qwen3/qwen3_gpu_patch_gen_config.py` "
            "for a reference pattern. Alternatively, set "
            "`model.ops_implementation.cross_entropy_loss_implementation` to `eager`."
        )
    return liger_kernel_cross_entropy(weights, hidden_states, labels), logits
