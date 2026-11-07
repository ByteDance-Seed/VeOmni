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
* **The counts all-reduce over every rank holding a distinct token slice**,
  which is ``fsdp_group``, the flattened ``dp_sp``
  (``dp_replicate x dp_shard x cp x ulysses``). Sequence-parallel ranks belong
  in it because ``slice_input_tensor`` gives each one a different slice of the
  sequence, so their routers score different tokens. **Expert-parallel ranks
  belong in it too**: the EP mesh is carved out of ``dp_shard_sp``, so EP peers
  are already inside this group, and each of them holds its own micro-batch.
  Tensor and pipeline parallelism are outside it, because TP peers score
  identical tokens and PP stages hold disjoint routers.

  Narrowing this group would be a silent correctness bug rather than a
  performance tweak. FSDP2 synchronises no buffers at all, so the bias is kept
  identical across ranks only by this reduce being global; a partial sum would
  leave ranks with divergent biases, routing differently, forever, with
  nothing raising.
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
        self._routers: list[nn.Module] = []
        self._seen: set[int] = set()
        self._counts: Optional[torch.Tensor] = None
        self._handles: list[Any] = []
        self._enabled = True

    def attach(self, model: nn.Module) -> int:
        """Hook every router that carries the bias buffer. Returns how many.

        ``model.modules()`` yields a deterministic order that is the same on
        every rank, since every rank builds the same model from the same
        config, so the row each router owns in the counts tensor needs no
        cross-rank agreement. Calling this twice is a no-op for routers already
        hooked, so a second call cannot silently double-count.
        """
        for module in model.modules():
            bias = getattr(module, BIAS_ATTR, None)
            if not isinstance(bias, torch.Tensor) or id(module) in self._seen:
                continue
            index = len(self._routers)
            self._seen.add(id(module))
            self._routers.append(module)
            self._handles.append(module.register_forward_hook(self._make_hook(index)))

        if self._routers:
            bias = getattr(self._routers[0], BIAS_ATTR)
            # One row per router, allocated up front rather than on first use:
            # a lazily built set of rows would differ between ranks whose
            # micro-batches reached different layers, and step() issues one
            # collective over the whole tensor.
            self._counts = torch.zeros((len(self._routers), bias.numel()), dtype=torch.long, device=bias.device)
        return len(self._routers)

    def _make_hook(self, index: int):
        def _hook(module: nn.Module, inputs, output):  # noqa: ANN001
            if not self._enabled or self._counts is None:
                return
            indices = _topk_indices_from_output(output)
            if indices is None:
                raise RuntimeError(
                    f"{type(module).__name__} carries '{BIAS_ATTR}' but its forward did not return "
                    "the (logits, weights, indices) tuple the bias update reads load from. "
                    "Failing loudly rather than silently freezing the bias: a router that stops "
                    "reporting its selection would leave load balancing switched off for the rest "
                    "of the run with nothing to show for it. Update _topk_indices_from_output in "
                    "veomni/utils/expert_bias.py for the new contract."
                )
            # scatter_add_ wants an int64 index and a src of the tensor's own
            # dtype; a router returning int32 indices would otherwise raise.
            flat = indices.detach().reshape(-1).to(device=self._counts.device, dtype=torch.long)
            self._counts[index].scatter_add_(0, flat, torch.ones_like(flat))

        return _hook

    @torch.no_grad()
    def step(self, group: Optional["ProcessGroup"] = None) -> None:
        """Update every router's bias from the load seen since the last call.

        Every rank must call this, and each one issues exactly one collective
        over the whole counts tensor however its own traffic fell out. A
        per-router reduce skipped on empty counts would make the number of
        collectives depend on local data, which deadlocks rather than erroring
        as soon as any rank disagrees. A router that genuinely saw no tokens
        contributes an all-zero row, whose sign is zero, so it is a no-op.
        """
        if self._counts is None:
            return

        load = self._counts.to(torch.float32)
        if group is not None and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM, group=group)

        # Underloaded experts (below their layer's mean) get a nudge up,
        # overloaded ones a nudge down, by a fixed step. torch.sign gives 0 for
        # an exactly-average expert, which correctly leaves it alone.
        direction = torch.sign(load.mean(dim=1, keepdim=True) - load)
        for index, module in enumerate(self._routers):
            bias = getattr(module, BIAS_ATTR)
            bias.add_(self.update_rate * direction[index].to(bias.dtype))
        self._counts.zero_()

    def disable(self) -> None:
        """Stop accumulating and release the hooks."""
        self._enabled = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._routers.clear()
        self._seen.clear()
        self._counts = None
