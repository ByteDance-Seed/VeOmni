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

"""OmniModel training-efficiency meter.

Split out from :mod:`veomni.utils.helper` because the OmniModel metric input is a
different shape from the single-model :class:`~veomni.utils.helper.EnvironMeter`:
FLOPs and token lengths are produced **per module** (each module's
:class:`~veomni.models.seed_omni.mixins.metric_meter_mixin.MetricMeterMixin`), so this meter does not
inspect the batch for token lengths at all.  It only owns the global,
module-agnostic concerns.
"""

import gc
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch.distributed as dist

from ..distributed.parallel_state import get_parallel_state
from . import logging
from .count_flops import get_device_flops
from .dist_utils import all_reduce
from .helper import (
    MultiSourceInfoTracker,
    _get_multisource_ds_idx,
    compute_device_memory_metrics,
    empty_cache,
)


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ..distributed.parallel_state import ParallelState

logger = logging.get_logger(__name__)


class OmniEnvironMeter:
    """Training-efficiency meter for OmniModel (per-module metrics + global roll-up).

    Unlike :class:`~veomni.utils.helper.EnvironMeter` — which counts tokens and
    estimates FLOPs itself from a single ``model_type`` — ``OmniModel`` is a
    *composition* of independent sub-modules with no single config to dispatch a
    FLOPs formula on.  So **FLOPs and token lengths are produced per module** by
    each module's :class:`~veomni.models.seed_omni.mixins.metric_meter_mixin.MetricMeterMixin` and
    handed to :meth:`step` as ``module_metrics``.

    This meter therefore does **not** inspect the batch for token lengths.  Its
    only jobs are:

    * :meth:`add` (per micro-batch) — count the number of training samples
      (``batch_count``) and gather the per-sample multi-source dataset indices.
      **No token-length computation here.**
    * :meth:`step` (per global step) — roll up the per-module metrics:
      **sum** the theoretical FLOPs (→ one overall MFU) but keep token statistics
      **per-module** (``trace/<module>/…``), since the backbone's tokens already
      include the other modules' tokens and merging would double-count. Reports
      the real sample count (from :meth:`add`) as the global chunk count; runs
      multi-source + memory.
    """

    def __init__(
        self,
        global_batch_size: int,
        enable_multisource: bool = False,
        dataloader: Optional["DataLoader"] = None,
        data_path: str = "",
        empty_cache_steps: int = 500,
        gc_steps: int = 0,
        parallel_state: Optional["ParallelState"] = None,
    ) -> None:
        self.global_batch_size = global_batch_size
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.enable_multisource = enable_multisource
        self.empty_cache_steps = empty_cache_steps
        self.gc_steps = gc_steps
        self.world_size = dist.get_world_size()
        # consume_tokens is per-module (the backbone's tokens already include the
        # text/image tokens the other modules also see, so a single merged total
        # would double-count); consume_chunks is the global real sample count.
        self.consume_tokens: Dict[str, int] = {}
        self.consume_chunks = 0
        # Per-step accumulators (reset in step), filled by add():
        self.batch_count = 0
        self.batch_ds_idx: List[int] = []

        if self.enable_multisource:
            if dataloader is None or data_path is None:
                raise ValueError(
                    "`dataloader` and `data_path` is required for `OmniEnvironMeter` with multi-source dataloader."
                )
            self.multisource_tracker = MultiSourceInfoTracker(
                dataloader=dataloader, data_path=data_path, parallel_state=self.parallel_state
            )

        if self.gc_steps > 0:
            gc.disable()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {"consume_tokens": self.consume_tokens, "consume_chunks": self.consume_chunks}
        if self.enable_multisource:
            state_dict.update({"multisource_tracker": self.multisource_tracker.state_dict()})
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.consume_tokens = state_dict["consume_tokens"]
        self.consume_chunks = state_dict["consume_chunks"]
        if self.enable_multisource:
            self.multisource_tracker.load_state_dict(state_dict["multisource_tracker"])

    def add(self, micro_batch: Dict[str, Any]) -> None:
        """Accumulate the sample count + multi-source dataset indices.

        Called once per micro-batch.  **Token lengths are NOT computed here** —
        they come from the modules' metric meters at :meth:`step`.  Here we only count
        the training samples (one per conversation) and, for multi-source, gather
        the per-sample dataset indices.
        """
        conversation_list = micro_batch.get("conversation_list")
        if conversation_list is not None:
            self.batch_count += len(conversation_list)
        if self.enable_multisource:
            self.batch_ds_idx.extend(_get_multisource_ds_idx(micro_batch))

    def step(
        self,
        delta_time: float,
        global_step: int,
        module_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Roll up per-module ``(theoretical_flops, seqlens)`` into metrics.

        ``module_metrics`` maps ``module_name → (theoretical_flops, seqlens)`` from
        every metered module (its time-independent contribution this step).

        * **FLOPs / MFU are summed across modules** — each module's FLOPs is a
          distinct compute (backbone layers vs ViT vs lm_head), so summing is
          correct and gives one overall MFU.
        * **Token statistics are per-module** — a token is *not* summable across
          modules: the backbone's per-sample lengths already include the text /
          image tokens the text-encoder / vision modules also count, so merging
          would double-count. Each module gets its own ``trace/<module>/`` tokens.

        Everything is DP-reduced in a single all-reduce, then the one whole-graph
        ``delta_time`` is applied. ``module_metrics`` must have the same keys on
        every DP rank; they are packed in sorted order, so insertion order may differ.
        """
        names = sorted(module_metrics)
        num_samples = len(self.batch_ds_idx)

        # Pack one DP all-reduce:
        # [total_flops, real_samples, num_ranks, tokens_m..., ranks_aligned_m...].
        total_flops_local = sum(flops for flops, _ in module_metrics.values())
        packed = [float(total_flops_local), float(self.batch_count), 1.0]
        packed += [float(sum(module_metrics[name][1])) for name in names]
        packed += [float(len(module_metrics[name][1]) == num_samples) for name in names]
        reduced = all_reduce(tuple(packed), op="sum", group=self.parallel_state.dp_group)

        total_flops = reduced[0]
        real_global_batch_size = int(reduced[1])
        num_ranks = int(reduced[2])
        global_tokens = dict(zip(names, reduced[3 : 3 + len(names)]))
        ranks_aligned = dict(zip(names, reduced[3 + len(names) :]))

        flops_achieved = total_flops / delta_time if delta_time else 0
        flops_promised = get_device_flops() * self.world_size
        mfu = flops_achieved / flops_promised if flops_promised else 0

        self.consume_chunks += real_global_batch_size

        metrics: Dict[str, Any] = {"consumed_chunk_num": self.consume_chunks}  # global real training samples
        # With no metered module the FLOPs are unmeasured, not zero.
        if names:
            metrics.update({"flops_achieved(T)": flops_achieved, "flops_promised(T)": flops_promised, "mfu": mfu})

        # Per-module token statistics (no cross-module merge → no double count).
        for name in names:
            tokens_m = global_tokens[name]
            self.consume_tokens[name] = self.consume_tokens.get(name, 0) + tokens_m
            prefix = f"trace/{name}/"
            metrics[prefix + "tokens_per_second(M)"] = tokens_m / delta_time / 1e6 if delta_time else 0
            metrics[prefix + "consume_tokens(M)"] = self.consume_tokens[name] / 1e6
            metrics[prefix + "consume_tokens(B)"] = self.consume_tokens[name] / 1e9
            # avg_seq_len: this module's tokens per configured sample slot (global_batch_size).
            metrics[prefix + "avg_seq_len"] = tokens_m / self.global_batch_size if self.global_batch_size else 0

        metrics.update(compute_device_memory_metrics())

        if self.enable_multisource:
            # The tracker step is a DP collective, so every rank enters it, and the
            # source module is chosen from reduced values so every rank picks the same one.
            source = self._multisource_module(global_tokens, ranks_aligned, num_ranks)
            if source is None:
                logger.warning_once(
                    "OmniEnvironMeter: no metered module reports per-sample seqlens aligned with ds_idx "
                    "on every DP rank; multi-source token counts are reported as zero."
                )
                per_sample_seqlens = [0] * num_samples
            else:
                per_sample_seqlens = module_metrics[source][1]
            metrics.update(self.multisource_tracker.step(self.batch_ds_idx, per_sample_seqlens))

        self.batch_count = 0
        self.batch_ds_idx = []

        if self.empty_cache_steps > 0 and global_step % self.empty_cache_steps == 0:
            empty_cache()

        if self.gc_steps > 0 and global_step % self.gc_steps == 0:
            gc.collect()

        return metrics

    @staticmethod
    def _multisource_module(
        global_tokens: Dict[str, float], ranks_aligned: Dict[str, float], num_ranks: int
    ) -> Optional[str]:
        """The module whose per-sample lengths feed multi-source accounting.

        Multi-source attributes the *whole training sequence* of each sample to
        its dataset, so we want the backbone's per-sample lengths (its packed
        sequence is the union of all modalities: text + image + boundary tokens).
        Candidates are modules with one length per sample on **every** DP rank;
        several can qualify (e.g. the text encoder and the backbone), so we take
        the one with the most global tokens, which is the backbone (a superset of
        the rest). Ties resolve to the first name in sorted order.
        """
        candidates = [name for name in sorted(global_tokens) if ranks_aligned[name] == num_ranks]
        return max(candidates, key=lambda name: global_tokens[name], default=None)


__all__ = ["OmniEnvironMeter"]
