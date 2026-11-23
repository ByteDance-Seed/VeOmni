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

"""Training hooks for :class:`DiffusionGemmaOmni`.

The loss side is the LLaDA families' contract: ``_loss`` is the local weighted sum,
``on_micro_batches_ready`` records the step's sample count, ``normalize_loss``
divides by the global total, and ``loss_is_step_mean`` tells the trainer not to
divide again. What differs is the batching.

**The batch is built by ``DiffusionGemmaCollator`` itself.** The collator
turns a list of per-sample dicts into the encoder row, the canvas stack and the
masks the decoder needs; packing (``flat_seq_pack``), the left-padded prompt path,
the ``compute_all_canvas_df_loss`` stack and the sequence-parallel tail alignment
all live in that one pure function. ``SeedOmniCollator`` only groups the carrier
items, so ``batch_pre`` re-instantiates the real collator and calls it. Copying its
logic here would be a second implementation of the canvas geometry, and the packing
branch is the part a copy most easily gets wrong.

Its ``sp_size`` and ``pack_samples`` are read from the live parallel state and the
model config rather than from the launcher, because the collator validates the pair
(``sp_size > 1`` requires packing) and a mismatch would otherwise surface as a
confusing shape error later.

What the collator cannot do is align the canvas axis *across* ranks: it pads to the
longest `K` inside one micro-batch, and ranks draw different samples. Under ``serial``
the decoder issues one pass per canvas, so an unequal `K` means unequal FSDP all-gathers
and expert-parallel all-to-alls, which hangs rather than computing a wrong number.
``batch_pre`` reduces the max `K` and pads, except under ``flat_seq_pack`` where a
padded slot belongs to no sample.
"""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from ......data.data_collator import DiffusionGemmaCollator, _local_canvas_count, pad_decoder_canvases
from ......distributed.parallel_state import get_parallel_state
from ......utils.dist_utils import all_reduce
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ...llada.accelerated.accelerated import TrainingMixin as LladaTrainingMixin
from ..modeling import DiffusionGemmaOmni


class TrainingMixin(LladaTrainingMixin):
    """Dense LLaDA's per-sample normalization, plus the canvas-aware batching."""

    #: Where this step sits, from :meth:`on_step_context`. `_self_conditioning_micro_step`
    #: counts micro-batches within it, so the self-conditioning draw keys on
    #: ``(seed, global_step, micro_step)`` -- a pure function of training position that a
    #: resume restores, rather than a generator a checkpoint cannot express.
    _self_conditioning_seed: int = 0
    _self_conditioning_global_step: int = 0
    _self_conditioning_micro_step: int = 0
    _self_conditioning_batch_invariant: bool = False

    def on_step_context(self, *, seed: int, global_step: int, batch_invariant: bool = False) -> None:
        """Start a new step's draw key, and reset the micro-batch counter."""
        self._self_conditioning_seed = int(seed)
        self._self_conditioning_global_step = int(global_step)
        self._self_conditioning_micro_step = 0
        self._self_conditioning_batch_invariant = bool(batch_invariant)

    def batch_pre(self, conversation_list: list[Any] = None, **kwargs: Any) -> dict[str, Any]:
        """Collate the carrier into the encoder/canvas batch the model reads.

        The carrier items already hold the fully built sample (the CPU preprocessor
        ran ``process_diffusion_gemma_example``), so this only has to gather those
        dicts and hand them to the real collator -- which owns packing, padding, the
        canvas stack and the decoder masks.

        The canvas axis is then aligned across ranks, which the collator cannot do:
        it pads to the longest `K` **in this micro-batch**, and different ranks draw
        different samples. Under ``serial`` the decoder issues one pass per canvas, so
        an unequal `K` means an unequal number of FSDP all-gathers and expert-parallel
        all-to-alls -- a hang, not a wrong number. ``flat_seq_pack`` is exempt because
        a padded slot there belongs to no sample and neither mask it could be given is
        legal; the mode also decodes in one pass, so no collective count depends on
        `K`. The branch is on the config, which is equal on every rank, so all ranks
        agree about reaching the reduction.
        """
        del kwargs
        if conversation_list is None:
            return {}

        features = []
        for sample in conversation_list:
            for item in sample:
                value = getattr(item, "value", None)
                if isinstance(value, dict) and "input_ids" in value:
                    features.append(value)
                    break
            else:
                raise ValueError(
                    "DiffusionGemmaOmni: no processed item in this sample. The module's "
                    "preprocessor must run before the forward; check that `data_type: seedomni` "
                    "and the module's `processing.py` are both wired."
                )

        state = get_parallel_state()
        mode = getattr(self.config, "canvas_batching", None)
        pack = bool(mode in ("flat_seq", "flat_seq_pack"))
        collator = DiffusionGemmaCollator(
            pad_token_id=int(getattr(self.config, "pad_token_id", 0) or 0),
            pack_samples=pack,
            sp_size=state.sp_size,
        )
        micro_batch = collator(features)

        if mode != "flat_seq_pack" and dist.is_available() and dist.is_initialized():
            target_k = int(all_reduce(_local_canvas_count(micro_batch["decoder_input_ids"]), op="max"))
            micro_batch = pad_decoder_canvases(micro_batch, target_k)

        # Every denoising step at inference hands the model its own previous prediction,
        # so training without that signal would meet it cold at generation. The draw is a
        # pure function of where training is (see `attach_self_conditioning`), and the
        # counter advances per micro-batch so gradient accumulation does not turn the
        # Bernoulli into a per-step coin.
        micro_batch = self.attach_self_conditioning(
            micro_batch,
            seed=self._self_conditioning_seed,
            global_step=self._self_conditioning_global_step,
            micro_step=self._self_conditioning_micro_step,
            batch_invariant=self._self_conditioning_batch_invariant,
        )
        self._self_conditioning_micro_step += 1
        return micro_batch

    @staticmethod
    def _micro_batch_sample_count(micro_batch: dict[str, Any]) -> int:
        """This micro-batch's supervised sample count: one per **example**, not per canvas.

        Runs on the *collated* batch, before ``batch_pre`` replaced the carrier with
        tensors, so the count comes from the items.

        A canvas sample can carry `K` decoder rows, and under
        ``compute_all_canvas_df_loss`` each row's weights are already divided by `K` so
        the rows sum to one example's mean. Counting rows would therefore divide that
        mean by `K` a second time, shrinking the whole step by a factor that tracks the
        answer-length distribution. So the count is "does this example contribute at
        all", which is also what the shared predicate decides for a draw that masked
        nothing: zero weights, still one sample of the mean.
        """
        conversation_list = micro_batch.get("conversation_list")
        if conversation_list is None:
            return 0
        count = 0
        for sample in conversation_list:
            for item in sample:
                value = getattr(item, "value", None)
                if not isinstance(value, dict) or "loss_weight" not in value:
                    continue
                if float(value["loss_weight"].sum()) > 0:
                    count += 1
                break
        return count


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class DiffusionGemmaOmniAccelerated(VeOmniMixin, DiffusionGemmaOmni):
    pass


__all__ = ["DiffusionGemmaOmniAccelerated"]
