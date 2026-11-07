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

"""Auxiliary-loss-free router balancing: the expert bias update.

DeepSeek-V3/V4 and GLM-MoE route with a per-expert bias term added to the
router scores before the top-k selection, and only for the selection: the
weight each chosen expert receives still comes from the unbiased score. The
strategy is from `Auxiliary-Loss-Free Load Balancing
<https://arxiv.org/abs/2408.15664>`_, and it only balances anything if
something moves that bias in response to observed load.

The routers declare the buffer and read it, but nothing wrote it, so the bias
stayed at whatever it was initialised or loaded with for the whole run. This
module supplies the missing half.

Design notes, in case they are not obvious:

* **Keyed on the buffer, not on a class name.** Any module carrying
  ``e_score_correction_bias`` is updated, so a future model gets this without
  touching a registry. That also keeps the change outside
  ``veomni/models/transformers/*/generated/``, which must not be hand-edited.
* **Counts are accumulated in a forward hook and consumed once per optimizer
  step**, not once per micro-batch, so gradient accumulation sees the load of
  the whole step. This is the same placement torchtitan uses.
* **The counts all-reduce over the group that holds distinct tokens.** Every
  rank's router scores its own token slice, and the bias must stay identical
  on all of them or they would route differently. Expert parallelism is not in
  that group: EP shards the experts, not the tokens, so EP peers already see
  the same routing decisions and summing over them would multiply the counts.
* **Update is sign-based**, as in the paper and in Megatron-LM and torchtitan:
  the bias moves by a fixed rate, not in proportion to the imbalance, which
  keeps it from reacting violently to one unusual step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from .logging import get_logger


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = get_logger(__name__)

BIAS_ATTR = "e_score_correction_bias"


def _topk_indices_from_output(output: Any) -> Optional[torch.Tensor]:
    """The selected experts, from a router's return value.

    All three routers that carry the bias today return
    ``(logits, weights, indices)``. Anything else is left alone rather than
    guessed at, so an unrecognised router disables itself instead of
    accumulating nonsense.
    """
    if isinstance(output, tuple) and len(output) == 3:
        indices = output[-1]
        if isinstance(indices, torch.Tensor) and indices.dtype in (torch.int32, torch.int64):
            return indices
    return None


class ExpertBiasUpdater:
    """Accumulates expert load and nudges the routers' correction bias.

    Args:
        update_rate: How far the bias moves per optimizer step. The paper's
            ``u``; Megatron-LM calls it ``--moe-router-bias-update-rate`` and
            torchtitan ``load_balance_coeff``.
    """

    def __init__(self, update_rate: float) -> None:
        if update_rate <= 0.0:
            raise ValueError(f"update_rate must be positive, got {update_rate}")
        self.update_rate = update_rate
        self._counts: dict[int, torch.Tensor] = {}
        self._routers: dict[int, nn.Module] = {}
        self._handles: list[Any] = []
        self._enabled = True

    def attach(self, model: nn.Module) -> int:
        """Hook every router that carries the bias buffer. Returns how many."""
        for module in model.modules():
            bias = getattr(module, BIAS_ATTR, None)
            if not isinstance(bias, torch.Tensor):
                continue
            key = id(module)
            self._routers[key] = module
            self._handles.append(module.register_forward_hook(self._make_hook(key)))
        return len(self._routers)

    def _make_hook(self, key: int):
        def _hook(module: nn.Module, inputs, output):  # noqa: ANN001
            if not self._enabled:
                return
            indices = _topk_indices_from_output(output)
            if indices is None:
                return
            bias = getattr(module, BIAS_ATTR)
            counts = self._counts.get(key)
            if counts is None:
                counts = torch.zeros(bias.numel(), dtype=torch.long, device=bias.device)
                self._counts[key] = counts
            flat = indices.detach().reshape(-1).to(counts.device)
            counts.scatter_add_(0, flat, torch.ones_like(flat))

        return _hook

    @torch.no_grad()
    def step(self, group: Optional["ProcessGroup"] = None) -> None:
        """Update every router's bias from the load seen since the last call.

        Must be called on every rank, because of the all-reduce. Routers that
        saw no tokens this step are skipped rather than driven by an empty
        count, which is what happens on a rank whose micro-batches did not
        reach a given layer.
        """
        for key, counts in self._counts.items():
            if counts.sum() == 0:
                continue
            load = counts.to(torch.float32)
            if group is not None and dist.is_initialized():
                dist.all_reduce(load, op=dist.ReduceOp.SUM, group=group)

            bias = getattr(self._routers[key], BIAS_ATTR)
            # Underloaded experts (below the mean) get a nudge up, overloaded
            # ones a nudge down, by a fixed step. torch.sign gives 0 for an
            # exactly-average expert, which correctly leaves it alone.
            direction = torch.sign(load.mean() - load)
            bias.add_(self.update_rate * direction.to(bias.dtype))
            counts.zero_()

    def disable(self) -> None:
        """Stop accumulating and release the hooks."""
        self._enabled = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._counts.clear()
        self._routers.clear()
