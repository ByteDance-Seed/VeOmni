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

"""OmniModel V2 training-efficiency meter.

Split out from :mod:`veomni.utils.helper` because the OmniModel trace is a
different shape from the single-model :class:`~veomni.utils.helper.EnvironMeter`:
FLOPs and token lengths are produced **per module** (each module's
:class:`~veomni.models.seed_omni.mixins.tracemixin.TraceMixin`), so this meter does not
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

logger = logging.get_logger(__name__)


class OmniEnvironMeter:
    """Training-efficiency meter for OmniModel V2 (per-module trace + global roll-up).

    Unlike :class:`~veomni.utils.helper.EnvironMeter` — which counts tokens and
    estimates FLOPs itself from a single ``model_type`` — ``OmniModel`` is a
    *composition* of independent sub-modules with no single config to dispatch a
    FLOPs formula on.  So **FLOPs and token lengths are produced per module** by
    each module's :class:`~veomni.models.seed_omni.mixins.tracemixin.TraceMixin` and
    handed to :meth:`step` as ``module_traces``.

    This meter therefore does **not** inspect the batch for token lengths.  Its
    only jobs are:

    * :meth:`add` (per micro-batch) — count the number of training samples
      (``batch_count``) and gather the per-sample multi-source dataset indices.
      **No token-length computation here.**
    * :meth:`step` (per global step) — roll up the per-module traces:
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
    ) -> None:
        self.global_batch_size = global_batch_size
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
            self.multisource_tracker = MultiSourceInfoTracker(dataloader=dataloader, data_path=data_path)

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
        they come from the modules' traces at :meth:`step`.  Here we only count
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
        module_traces: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Roll up per-module ``(theoretical_flops, seqlens)`` into the trace.

        ``module_traces`` maps ``module_name → (theoretical_flops, seqlens)`` from
        every tracing module (its time-independent contribution this step).

        * **FLOPs / MFU are summed across modules** — each module's FLOPs is a
          distinct compute (backbone layers vs ViT vs lm_head), so summing is
          correct and gives one overall MFU.
        * **Token statistics are per-module** — a token is *not* summable across
          modules: the backbone's per-sample lengths already include the text /
          image tokens the text-encoder / vision modules also count, so merging
          would double-count. Each module gets its own ``trace/<module>/`` tokens.

        Everything is DP-reduced in a single all-reduce, then the one whole-graph
        ``delta_time`` is applied.
        """
        names = list(module_traces.keys())

        # Pack one DP all-reduce: [total_flops, real_samples, (tokens_m, chunks_m)*].
        total_flops_local = sum(flops for flops, _ in module_traces.values())
        packed = [float(total_flops_local), float(self.batch_count)]
        for name in names:
            _flops, seqlens = module_traces[name]
            packed.append(float(sum(seqlens)))  # tokens for this module
        reduced = all_reduce(tuple(packed), op="sum", group=get_parallel_state().dp_group)
        if not isinstance(reduced, list):  # single-element edge case
            reduced = [reduced]

        total_flops = reduced[0]
        real_global_batch_size = int(reduced[1])

        flops_achieved = total_flops / delta_time if delta_time else 0
        flops_promised = get_device_flops() * self.world_size
        mfu = flops_achieved / flops_promised if flops_promised else 0

        self.consume_chunks += real_global_batch_size

        metrics: Dict[str, Any] = {
            "flops_achieved(T)": flops_achieved,
            "flops_promised(T)": flops_promised,
            "mfu": mfu,
            "consumed_chunk_num": self.consume_chunks,  # global real training samples
        }

        # Per-module token statistics (no cross-module merge → no double count).
        for i, name in enumerate(names):
            tokens_m = reduced[2 + i]
            self.consume_tokens[name] = self.consume_tokens.get(name, 0) + tokens_m
            prefix = f"trace/{name}/"
            metrics[prefix + "tokens_per_second(M)"] = tokens_m / delta_time / 1e6 if delta_time else 0
            metrics[prefix + "consume_tokens(M)"] = self.consume_tokens[name] / 1e6
            metrics[prefix + "consume_tokens(B)"] = self.consume_tokens[name] / 1e9
            # avg_seq_len: this module's tokens per configured sample slot (global_batch_size).
            metrics[prefix + "avg_seq_len"] = tokens_m / self.global_batch_size if self.global_batch_size else 0

        metrics.update(compute_device_memory_metrics())

        if self.enable_multisource:
            # Multi-source needs one token length per sample; use the module whose
            # seqlens are per-sample (the backbone: len == sample count). Other
            # modules report per-image / single-chunk lengths that don't align.
            per_sample_seqlens = self._per_sample_seqlens(module_traces, len(self.batch_ds_idx))
            if per_sample_seqlens is not None:
                metrics.update(self.multisource_tracker.step(self.batch_ds_idx, per_sample_seqlens))
            else:
                logger.warning_once(
                    "OmniEnvironMeter: multi-source accounting skipped — no traced module reports "
                    f"per-sample seqlens aligned with ds_idx ({len(self.batch_ds_idx)} samples)."
                )

        self.batch_count = 0
        self.batch_ds_idx = []

        if self.empty_cache_steps > 0 and global_step % self.empty_cache_steps == 0:
            empty_cache()

        if self.gc_steps > 0 and global_step % self.gc_steps == 0:
            gc.collect()

        return metrics

    @staticmethod
    def _per_sample_seqlens(module_traces: Dict[str, Any], num_samples: int) -> Optional[List[int]]:
        """Canonical per-sample token lengths for multi-source accounting.

        Multi-source attributes the *whole training sequence* of each sample to
        its dataset, so we want the backbone's per-sample lengths (its packed
        sequence is the union of all modalities — text + image + boundary
        tokens). Several modules may have one entry per sample (e.g. both the
        text encoder and the backbone), so picking by length alone is ambiguous;
        among the per-sample-aligned modules we take the one with the **most
        tokens**, which is always the backbone (a superset of the rest).
        """
        if num_samples <= 0:
            return None
        best: Optional[List[int]] = None
        best_total = -1
        for _theoretical_flops, seqlens in module_traces.values():
            if len(seqlens) == num_samples:
                total = sum(seqlens)
                if total > best_total:
                    best_total = total
                    best = seqlens
        return best


__all__ = ["OmniEnvironMeter"]
