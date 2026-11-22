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

**The batch is built by the non-Omni collator itself.** ``DiffusionGemmaCollator``
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
"""

from __future__ import annotations

from typing import Any

from ......data.data_collator import DiffusionGemmaCollator
from ......distributed.parallel_state import get_parallel_state
from ......utils.loss_utils import count_supervised_samples
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ...llada.accelerated.accelerated import TrainingMixin as LladaTrainingMixin
from ..modeling import DiffusionGemmaOmni


class TrainingMixin(LladaTrainingMixin):
    """Dense LLaDA's per-sample normalization, plus the canvas-aware batching."""

    def batch_pre(self, conversation_list: list[Any] = None, **kwargs: Any) -> dict[str, Any]:
        """Collate the carrier into the encoder/canvas batch the model reads.

        The carrier items already hold the fully built sample (the CPU preprocessor
        ran ``process_diffusion_gemma_example``), so this only has to gather those
        dicts and hand them to the real collator -- which owns packing, padding, the
        canvas stack and the decoder masks.
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
        pack = bool(getattr(self.config, "canvas_batching", None) in ("flat_seq", "flat_seq_pack"))
        collator = DiffusionGemmaCollator(
            pad_token_id=int(getattr(self.config, "pad_token_id", 0) or 0),
            pack_samples=pack,
            sp_size=state.sp_size,
        )
        return collator(features)

    @staticmethod
    def _micro_batch_sample_count(micro_batch: dict[str, Any]) -> int:
        """This micro-batch's supervised sample count.

        Runs on the *collated* batch, before ``batch_pre`` replaced the carrier with
        tensors, so the count comes from the items. It delegates to the shared
        predicate so a draw whose canvas masked nothing still counts as one sample of
        the mean, exactly as on the dense LLaDA path.
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
                count += count_supervised_samples(value)
                break
        return count


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class DiffusionGemmaOmniAccelerated(VeOmniMixin, DiffusionGemmaOmni):
    pass


__all__ = ["DiffusionGemmaOmniAccelerated"]
