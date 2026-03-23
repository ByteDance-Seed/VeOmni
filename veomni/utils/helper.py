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


"""Helper utils"""

import datetime
import gc
import logging as builtin_logging
import os
import random
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers import set_seed as set_seed_func

from ..data.batch_metadata import get_batch_metadata, pop_batch_metadata
from ..distributed.parallel_state import get_parallel_state
from . import logging
from .count_flops import VeomniFlopsCounter
from .device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_torch_device,
)
from .dist_utils import all_reduce
from .multisource_utils import parse_multisource_config
from .seqlen_pos_transform_utils import valid_seqlens_from_cu_seqlens


try:
    import hdfs_io
    from hdfs_io import copy
except (ImportError, ModuleNotFoundError):
    from veomni.utils import hdfs_io
    from veomni.utils.hdfs_io import copy

if IS_NPU_AVAILABLE:
    import torch_npu


# internal use
VALID_CONFIG_TYPE = None
VEOMNI_UPLOAD_CMD = None
FlopsCounter = None


def convert_hdfs_fuse_path(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    return kwargs.get("path", None)


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))


def _compute_seqlens(micro_batch: Dict[str, "torch.Tensor"]) -> List[int]:
    if "cu_seq_lens_q" in micro_batch:
        # packed micro batch
        seqlens = valid_seqlens_from_cu_seqlens(micro_batch["cu_seq_lens_q"]).tolist()
        return seqlens
    elif "attention_mask" in micro_batch:
        # unpacked sample
        attention_mask = micro_batch["attention_mask"]
        seqlens = attention_mask.sum().item()
        return [seqlens]
    elif "chosen_attention_mask" in micro_batch:
        # DPO preference pair — report combined chosen + rejected length
        chosen_len = micro_batch["chosen_attention_mask"].sum().item()
        rejected_len = micro_batch["rejected_attention_mask"].sum().item()
        return [chosen_len + rejected_len]
    else:
        return [0]


def _compute_image_seqlens(micro_batch: Dict[str, "torch.Tensor"]) -> List[int]:
    image_shape_keys = ["image_grid_thw", "video_grid_thw", "audio_grid_thw"]
    image_seqlens = []
    for key in image_shape_keys:
        if key in micro_batch:
            grid_thw = micro_batch[key]
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
            image_seqlens.extend(seqlens)
    return image_seqlens


def _compute_wan_seqlens(micro_batch: Dict[str, "torch.Tensor"]) -> List[int]:
    dit_latents_seqlens = []
    for latents in micro_batch["latents"]:
        latent_shape = latents.shape
        if len(latent_shape) == 5:
            B = latent_shape[0]
        else:
            B = 1
        C, T, H, W = latent_shape[-4:]
        T_out = int((T - 1) / 1 + 1)
        H_out = int((H - 2) / 2 + 1)
        W_out = int((W - 2) / 2 + 1)
        seqlens = B * T_out * H_out * W_out
        dit_latents_seqlens.append(seqlens)
    return dit_latents_seqlens


def _get_multisource_ds_idx(micro_batch: Dict[str, "torch.Tensor"]) -> List[int]:
    """Extract per-sample ds_idx list from a micro-batch.

    Reads from ``_batch_metadata`` first (new collator path).  Falls back to
    popping ``ds_idx`` from the top-level dict for backward compatibility with
    collators that have not yet adopted the metadata mechanism.
    """
    meta = get_batch_metadata(micro_batch)
    ds_idx = meta.get("ds_idx")
    if ds_idx is not None:
        # New path: ds_idx is already a List[int] from PackingCollator.
        if isinstance(ds_idx, list):
            return ds_idx
        if isinstance(ds_idx, torch.Tensor):
            return ds_idx.tolist()
        return [ds_idx]

    # Backward compatibility: ds_idx at top level (old collator).
    ds_idx = micro_batch.pop("ds_idx", None)
    micro_batch.pop("source_name", None)
    micro_batch.pop("cur_token_num", None)
    if ds_idx is None:
        return []
    if isinstance(ds_idx, torch.Tensor):
        return ds_idx.tolist()
    return [ds_idx]


def _percentile_from_hist(hist: "torch.Tensor", bins: "torch.Tensor", q: float) -> float:
    """Approximate a percentile from a histogram via linear interpolation."""
    cdf = hist.cumsum(0)
    total = cdf[-1].item()
    if total == 0:
        return 0.0
    target = total * q
    idx = torch.searchsorted(cdf, target).clamp(max=len(bins) - 2).item()
    lower = cdf[idx - 1].item() if idx > 0 else 0.0
    upper = cdf[idx].item()
    frac = (target - lower) / max(upper - lower, 1e-9)
    return bins[idx].item() + frac * (bins[idx + 1].item() - bins[idx].item())


class EnvironMeter:
    """
    Computes the metrics about the training efficiency.

    Args:
        config (PretrainedConfig): The configuration of the model.
        global_batch_size (int): The global batch size.
        enable_multisource (bool, optional): Whether to enable the multi-source dataloader. Defaults to False.
        dataloader (DataLoader, optional): The training dataloader for multi-source dataloader. Defaults to None.
        data_path (str, optional): The data path for multi-source dataloader. Defaults to "".
        empty_cache_steps (int, optional): The number of steps to empty the cache. Defaults to 500.
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        global_batch_size: int,
        enable_multisource: bool = False,
        dataloader: Optional["DataLoader"] = None,
        data_path: str = "",
        empty_cache_steps: int = 500,
        gc_steps: int = 0,
        track_seqlen_distribution: bool = True,
        seqlen_distribution_log_interval: int = 10,
    ) -> None:
        self.config = config
        self.global_batch_size = global_batch_size
        self.enable_multisource = enable_multisource
        self.empty_cache_steps = empty_cache_steps
        self.gc_steps = gc_steps
        self.world_size = dist.get_world_size()
        self.consume_tokens = 0
        self.consume_chunks = 0
        self.batch_seqlens = []
        self.batch_ds_idx = []
        self.images_seqlens = []
        # Throughput tracking
        self.batch_samples = 0
        self.batch_bytes = 0
        self.consume_samples = 0
        self.consume_bytes = 0

        # Sequence length distribution tracking
        self.track_seqlen_distribution = track_seqlen_distribution
        self.seqlen_distribution_log_interval = seqlen_distribution_log_interval

        # Histogram-based seqlen tracking
        self._seqlen_bins = torch.tensor(
            [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
            dtype=torch.float64,
        )
        self._seqlen_hist_local = torch.zeros(len(self._seqlen_bins) - 1, dtype=torch.float64)
        self._seqlen_local_stats = torch.tensor([0.0, 0.0, float("inf"), float("-inf")], dtype=torch.float64)

        if self.enable_multisource:
            if dataloader is None or data_path is None:
                raise ValueError(
                    "`dataloader` and `data_path` is required for `EnvironMeter` with multi-source dataloader."
                )

            self.multisource_tracker = MultiSourceInfoTracker(dataloader=dataloader, data_path=data_path)

        # for internal use
        if VALID_CONFIG_TYPE is not None and isinstance(config, VALID_CONFIG_TYPE):
            self.estimate_flops = FlopsCounter(config).estimate_flops
        else:
            self.estimate_flops = VeomniFlopsCounter(config).estimate_flops

        if self.gc_steps > 0:
            gc.disable()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {
            "consume_tokens": self.consume_tokens,
            "consume_chunks": self.consume_chunks,
            "consume_samples": self.consume_samples,
            "consume_bytes": self.consume_bytes,
        }
        if self.enable_multisource:
            state_dict.update({"multisource_tracker": self.multisource_tracker.state_dict()})

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.consume_tokens = state_dict["consume_tokens"]
        self.consume_chunks = state_dict["consume_chunks"]
        self.consume_samples = state_dict.get("consume_samples", 0)
        self.consume_bytes = state_dict.get("consume_bytes", 0)
        if self.enable_multisource:
            self.multisource_tracker.load_state_dict(state_dict["multisource_tracker"])

    def add(self, micro_batch: Union[Dict[str, "torch.Tensor"], List[Dict[str, "torch.Tensor"]]]) -> None:
        prev_len = len(self.batch_seqlens)
        if getattr(self.config, "condition_model_type", None) is None:  # hf model
            if isinstance(micro_batch, List):
                for sample in micro_batch:
                    self.batch_seqlens.extend(_compute_seqlens(sample))
                    self.images_seqlens.extend(_compute_image_seqlens(sample))
                    if self.enable_multisource:
                        self.batch_ds_idx.extend(_get_multisource_ds_idx(sample))
            else:
                self.batch_seqlens.extend(_compute_seqlens(micro_batch))
                self.images_seqlens.extend(_compute_image_seqlens(micro_batch))
                if self.enable_multisource:
                    self.batch_ds_idx.extend(_get_multisource_ds_idx(micro_batch))
        else:  # dit diffusers model
            self.batch_seqlens.extend(_compute_wan_seqlens(micro_batch))
        new_seqlens = self.batch_seqlens[prev_len:]

        # Track sequence lengths for distribution statistics (only new seqlens from this add)
        if self.track_seqlen_distribution and len(new_seqlens) > 0:
            seqlens_t = torch.tensor(new_seqlens, dtype=torch.float64)
            hist = torch.histogram(seqlens_t, bins=self._seqlen_bins)
            self._seqlen_hist_local += hist.hist
            self._seqlen_local_stats[0] += seqlens_t.sum()
            self._seqlen_local_stats[1] += len(new_seqlens)
            self._seqlen_local_stats[2] = min(self._seqlen_local_stats[2].item(), seqlens_t.min().item())
            self._seqlen_local_stats[3] = max(self._seqlen_local_stats[3].item(), seqlens_t.max().item())

        # Track samples and bytes
        _CORE_KEYS = {"input_ids", "labels", "attention_mask", "position_ids"}
        if isinstance(micro_batch, list):
            num_samples = len(micro_batch)
            num_bytes = sum(
                sum(v.nbytes for k, v in s.items() if k in _CORE_KEYS and isinstance(v, torch.Tensor))
                for s in micro_batch
            )
        else:
            num_samples = max(len(new_seqlens), 1)
            num_bytes = sum(
                v.nbytes for k, v in micro_batch.items() if k in _CORE_KEYS and isinstance(v, torch.Tensor)
            )
        self.batch_samples += num_samples
        self.batch_bytes += num_bytes

    def step(self, delta_time: float, global_step: int) -> Dict[str, Any]:
        if len(self.images_seqlens) > 0:
            flops_achieved, flops_promised = self.estimate_flops(
                self.batch_seqlens, delta_time, images_seqlens=self.images_seqlens
            )
        else:
            flops_achieved, flops_promised = self.estimate_flops(self.batch_seqlens, delta_time)
        batch_samples = self.batch_samples
        batch_bytes = self.batch_bytes

        flops_achieved, batch_tokens, real_global_batch_size, global_batch_samples, global_batch_bytes = all_reduce(
            (flops_achieved, sum(self.batch_seqlens), len(self.batch_seqlens), batch_samples, batch_bytes),
            op="sum",
            group=get_parallel_state().dp_group,
        )
        flops_promised = flops_promised * self.world_size
        mfu = flops_achieved / flops_promised

        # calculate average effective len and tokens per second
        avg_effective_len = batch_tokens / self.global_batch_size
        avg_sample_seq_len = batch_tokens / real_global_batch_size
        tokens_per_second = batch_tokens / delta_time
        self.consume_tokens += batch_tokens
        self.consume_chunks += real_global_batch_size

        samples_per_second = global_batch_samples / delta_time if delta_time > 0 else 0.0
        mb_per_second = (global_batch_bytes / (1024 * 1024)) / delta_time if delta_time > 0 else 0.0
        self.consume_samples += global_batch_samples
        self.consume_bytes += global_batch_bytes

        # cuda memory
        allocated_memory = get_torch_device().max_memory_allocated()
        reserved_memory = get_torch_device().max_memory_reserved()
        num_alloc_retries = get_torch_device().memory_stats()["num_alloc_retries"]
        allocated_memory, reserved_memory, num_alloc_retries = all_reduce(
            (allocated_memory, reserved_memory, num_alloc_retries), op="max"
        )

        # cpu memory
        cpu_memory_info = psutil.virtual_memory()

        metrics = {
            "flops_achieved(T)": flops_achieved,
            "flops_promised(T)": flops_promised,
            "mfu": mfu,
            "training/avg_effective_len": avg_effective_len,
            "training/avg_sample_seq_len": avg_sample_seq_len,
            "tokens_per_second(M)": tokens_per_second / 1e6,
            "consume_tokens(M)": self.consume_tokens / 1e6,
            "consume_tokens(B)": self.consume_tokens / 1e9,
            "consumed_chunk_num": self.consume_chunks,
            "samples_per_second": samples_per_second,
            "MB_per_second": mb_per_second,
            "consume_samples": self.consume_samples,
            "consume_bytes(GB)": self.consume_bytes / (1024**3),
            "max_memory_allocated(GB)": allocated_memory / (1024**3),
            "max_memory_reserved(GB)": reserved_memory / (1024**3),
            "cpu_used_memory(GB)": cpu_memory_info.used / (1024**3),
            "cpu_available_memory(GB)": cpu_memory_info.available / (1024**3),
            "cpu_memory_usage(%)": cpu_memory_info.percent,
            "num_alloc_retries": num_alloc_retries,
        }

        if self.enable_multisource:
            metrics.update(self.multisource_tracker.step(self.batch_ds_idx, self.batch_seqlens))

        # Sequence length distribution statistics
        if self.track_seqlen_distribution and self._seqlen_local_stats[1].item() > 0:
            if global_step % self.seqlen_distribution_log_interval == 0:
                dp_group = get_parallel_state().dp_group
                device = torch.device(get_device_type())
                global_hist = self._seqlen_hist_local.to(device)
                dist.all_reduce(global_hist, op=dist.ReduceOp.SUM, group=dp_group)
                stats_sum = self._seqlen_local_stats[:2].clone().to(device)
                dist.all_reduce(stats_sum, op=dist.ReduceOp.SUM, group=dp_group)
                stats_min = self._seqlen_local_stats[2:3].clone().to(device)
                dist.all_reduce(stats_min, op=dist.ReduceOp.MIN, group=dp_group)
                stats_max = self._seqlen_local_stats[3:4].clone().to(device)
                dist.all_reduce(stats_max, op=dist.ReduceOp.MAX, group=dp_group)
                total_count = stats_sum[1].item()
                if total_count > 0:
                    global_hist_cpu = global_hist.cpu()
                    metrics.update(
                        {
                            "seqlen/mean": stats_sum[0].item() / total_count,
                            "seqlen/min": stats_min[0].item(),
                            "seqlen/max": stats_max[0].item(),
                            "seqlen/median": _percentile_from_hist(global_hist_cpu, self._seqlen_bins, 0.50),
                            "seqlen/p95": _percentile_from_hist(global_hist_cpu, self._seqlen_bins, 0.95),
                            "seqlen/p99": _percentile_from_hist(global_hist_cpu, self._seqlen_bins, 0.99),
                        }
                    )
                self._seqlen_hist_local.zero_()
                self._seqlen_local_stats = torch.tensor([0.0, 0.0, float("inf"), float("-inf")], dtype=torch.float64)

        if self.empty_cache_steps > 0 and global_step % self.empty_cache_steps == 0:
            empty_cache()

        if self.gc_steps > 0 and global_step % self.gc_steps == 0:
            gc.collect()

        self.batch_seqlens = []
        self.batch_ds_idx = []
        self.images_seqlens = []
        self.batch_samples = 0
        self.batch_bytes = 0

        return metrics


@dataclass
class MultiSourceCounterItem:
    num_tokens: int = 0
    num_samples: int = 0
    num_steps: int = 0

    def increment(self, num_tokens: int, num_samples: int) -> None:
        self.num_tokens += num_tokens
        self.num_samples += num_samples

    def step(self) -> None:
        self.num_steps += 1


class MultiSourceInfoTracker:
    """
    Tracks the statistics about the MultiSourceDataset.
    """

    def __init__(self, dataloader: Optional["DataLoader"], data_path: str) -> None:
        self.dataloader = dataloader
        self.accumulate_counter = dict()
        self.batch_idx = 0
        self.multisource_config = parse_multisource_config(data_path)
        self.names = self.multisource_config["names"]
        self.boundary_type = self.multisource_config.get("boundary_type", "token")

    def state_dict(self) -> Dict[str, Any]:
        return {"accumulate_counter": self.accumulate_counter, "batch_idx": self.batch_idx}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.accumulate_counter = state_dict["accumulate_counter"]
        self.batch_idx = state_dict["batch_idx"]

    def step(self, batch_ds_idx: List[int], batch_seqlens: List[int]) -> Dict[str, Any]:
        """
        Computes the statistics about the MultiSourceDataset. It should be called at every rank to update dataloader.
        """
        counter = defaultdict(MultiSourceCounterItem)
        for ds_idx, seq_len in zip(batch_ds_idx, batch_seqlens):
            counter[ds_idx].increment(seq_len, 1)

        counter_list: List[Dict[int, MultiSourceCounterItem]] = [None for _ in range(get_parallel_state().dp_size)]
        dist.all_gather_object(counter_list, counter, group=get_parallel_state().dp_group)

        global_counter = defaultdict(MultiSourceCounterItem)
        for counter in counter_list:
            for ds_idx, item in counter.items():
                global_counter[ds_idx].increment(item.num_tokens, item.num_samples)
                self.accumulate_counter.setdefault(ds_idx, MultiSourceCounterItem()).increment(
                    item.num_tokens, item.num_samples
                )

        step_consumed_tokens = sum([item.num_tokens for item in global_counter.values()])
        global_consumed_tokens = sum([item.num_tokens for item in self.accumulate_counter.values()])
        step_consumed_samples = sum([item.num_samples for item in global_counter.values()])
        global_comsumed_samples = sum([item.num_samples for item in self.accumulate_counter.values()])

        if hasattr(self.dataloader, "update_consumed_tokens") and (
            not get_parallel_state().tp_enabled or get_parallel_state().tp_rank == 0
        ):  # update at every dp rank
            if self.boundary_type == "token":
                self.dataloader.update_consumed_tokens((self.batch_idx, global_consumed_tokens))
            elif self.boundary_type == "sample":
                self.dataloader.update_consumed_tokens((self.batch_idx, global_comsumed_samples))

        self.batch_idx += 1
        multisource_info = {}
        for ds_idx, item in self.accumulate_counter.items():
            multisource_info.update(
                {
                    "multi_source/global_consumed_tokens": global_consumed_tokens,
                    "multi_source/step_consumed_tokens": step_consumed_tokens,
                    "multi_source/global_consumed_samples": global_comsumed_samples,
                    "multi_source/step_consumed_samples": step_consumed_samples,
                    f"multi_source/consumed_chunk_num/{self.names[ds_idx]}": self.accumulate_counter[
                        ds_idx
                    ].num_samples,
                    f"multi_source/step_consumed_chunk_num/{self.names[ds_idx]}": global_counter[ds_idx].num_samples,
                    f"multi_source/consume_tokens(M)/{self.names[ds_idx]}": self.accumulate_counter[ds_idx].num_tokens
                    / 1e6,
                    f"multi_source/estimated_avg_chunk_len/{self.names[ds_idx]}": self.accumulate_counter[
                        ds_idx
                    ].num_tokens
                    / max(self.accumulate_counter[ds_idx].num_samples, 1),
                    f"multi_source/step_consumed_tokens(M)/{self.names[ds_idx]}": global_counter[ds_idx].num_tokens
                    / 1e6,
                    f"multi_source/step_consumed_ratio/{self.names[ds_idx]}": global_counter[ds_idx].num_tokens
                    / step_consumed_tokens,
                }
            )

        return multisource_info


class PerSourceLossTracker:
    """
    Tracks per-source loss by periodically sampling and recomputing loss.

    This tracker samples microbatches at specified intervals and recomputes
    the loss with torch.no_grad() to get per-source breakdown.

    Features:
    - Handles packed multi-source sequences accurately via source_boundaries
    - Proper error logging and exception handling
    - Configurable distributed logging (rank 0 only by default)

    Args:
        sample_interval: Compute per-source loss every N steps (default: 100)
        log_rank: Which rank should log (default: 0, only rank 0). Set to -1 for all ranks.
        global_rank: The global rank of this process.
        source_names: List of source names indexed by ds_idx.
    """

    def __init__(
        self,
        sample_interval: int = 100,
        log_rank: int = 0,
        global_rank: int = 0,
        source_names: Optional[List[str]] = None,
    ) -> None:
        self.sample_interval = sample_interval
        self.log_rank = log_rank
        self.global_rank = global_rank
        self.source_names = source_names or []

        # State for sampling
        self._sampled_batch: Optional[Dict[str, "torch.Tensor"]] = None
        self._source_metadata: List[Dict[str, Any]] = []
        self._has_captured = False  # Flag to avoid redundant should_sample() checks

    def should_sample(self, global_step: int) -> bool:
        """Check if we should sample at this step."""
        return global_step % self.sample_interval == 0

    def capture_batch(
        self,
        micro_batch: Dict[str, "torch.Tensor"],
        enable_multisource: bool = False,
    ) -> None:
        """
        Capture a microbatch for per-source loss computation.

        Args:
            micro_batch: The microbatch to potentially sample.
            enable_multisource: Whether multisource tracking is enabled.
        """
        if not enable_multisource:
            return

        # Extract source metadata
        self._source_metadata = self._extract_source_info(micro_batch)

        if not self._source_metadata:
            return

        # Store a copy of inputs for recompute
        self._sampled_batch = {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
            if k in ["input_ids", "attention_mask", "position_ids", "labels"]
        }
        self._has_captured = True

    def has_pending_computation(self) -> bool:
        """Check if there is a captured batch pending loss computation."""
        return self._has_captured

    def _extract_source_info(self, micro_batch: Dict[str, "torch.Tensor"]) -> List[Dict[str, Any]]:
        """
        Extract source information with accurate boundaries.

        Returns list of dicts with keys:
        - ds_idx: int
        - source_name: str
        - start_pos: int (in packed sequence)
        - end_pos: int (in packed sequence)
        """
        # Check for source_boundaries in _batch_metadata (new path)
        meta = get_batch_metadata(micro_batch)
        if "source_boundaries" in meta:
            return meta["source_boundaries"]

        # Backward compatibility: check top-level key
        if "source_boundaries" in micro_batch:
            return micro_batch["source_boundaries"]

        # Fallback: use ds_idx from metadata or top level (simplified, less accurate)
        ds_idx_tensor = meta.get("ds_idx") or micro_batch.get("ds_idx")
        if ds_idx_tensor is None:
            return []

        # Handle different formats
        if isinstance(ds_idx_tensor, torch.Tensor):
            if ds_idx_tensor.dim() == 0:
                # Single scalar
                ds_idx = ds_idx_tensor.item()
                seqlen = micro_batch["input_ids"].shape[-1]
                return [
                    {
                        "ds_idx": ds_idx,
                        "source_name": self._get_source_name(ds_idx),
                        "start_pos": 0,
                        "end_pos": seqlen,
                    }
                ]
            else:
                # Tensor of indices - use boundaries from unique consecutive values
                ds_idx_list = ds_idx_tensor.tolist()
                boundaries = []
                current_idx = ds_idx_list[0]
                start_pos = 0

                for i, idx in enumerate(ds_idx_list):
                    if idx != current_idx:
                        boundaries.append(
                            {
                                "ds_idx": current_idx,
                                "source_name": self._get_source_name(current_idx),
                                "start_pos": start_pos,
                                "end_pos": i,
                            }
                        )
                        current_idx = idx
                        start_pos = i

                # Add last segment
                boundaries.append(
                    {
                        "ds_idx": current_idx,
                        "source_name": self._get_source_name(current_idx),
                        "start_pos": start_pos,
                        "end_pos": len(ds_idx_list),
                    }
                )

                return boundaries
        else:
            # Single integer
            ds_idx = ds_idx_tensor
            seqlen = micro_batch["input_ids"].shape[-1]
            return [
                {
                    "ds_idx": ds_idx,
                    "source_name": self._get_source_name(ds_idx),
                    "start_pos": 0,
                    "end_pos": seqlen,
                }
            ]

    def _get_source_name(self, ds_idx: int) -> str:
        """Get source name from dataset index."""
        if 0 <= ds_idx < len(self.source_names):
            return self.source_names[ds_idx]
        return f"source_{ds_idx}"

    def compute_per_source_loss(
        self,
        model: "nn.Module",
    ) -> Dict[str, float]:
        """
        Recompute loss per source using torch.no_grad().

        Args:
            model: The model to use for forward pass.

        Returns:
            Dict mapping metric names to loss values.
        """
        if self._sampled_batch is None or not self._source_metadata:
            return {}

        per_source_metrics = {}
        source_losses: Dict[str, List[float]] = defaultdict(list)

        try:
            # --- GPU memory pre-check ---
            # Estimate memory needed: logits tensor is the dominant cost
            # logits shape = [1, seq_len, vocab_size], dtype = model's param dtype
            try:
                model_device = next(model.parameters()).device
                param_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_device = torch.device(get_device_type())
                param_dtype = torch.float16

            if model_device.type in ("cuda", "npu"):
                seq_len = self._sampled_batch.get("input_ids", torch.empty(0)).shape[-1] if self._sampled_batch else 0
                # Estimate: logits(seq_len * vocab_size * dtype_bytes) + activations (~2x logits)
                bytes_per_element = 2 if param_dtype in (torch.float16, torch.bfloat16) else 4
                # Use model config vocab_size if available, otherwise use a conservative default
                vocab_size = getattr(getattr(model, "config", None), "vocab_size", 152000)
                estimated_bytes = seq_len * vocab_size * bytes_per_element * 3  # logits + ~2x overhead
                if hasattr(torch, get_device_type()) and hasattr(getattr(torch, get_device_type()), "mem_get_info"):
                    free_mem, _ = getattr(torch, get_device_type()).mem_get_info()
                elif model_device.type == "cuda":
                    free_mem, _ = torch.cuda.mem_get_info()
                else:
                    free_mem = float("inf")  # Cannot check, proceed optimistically

                if estimated_bytes > 0 and free_mem < estimated_bytes * 1.2:
                    if self.global_rank == 0 or self.log_rank == -1:
                        logger.info(
                            f"Skipping per-source loss: insufficient GPU memory "
                            f"(free={free_mem / 1024**3:.1f}GB, "
                            f"estimated_need={estimated_bytes * 1.2 / 1024**3:.1f}GB, "
                            f"seq_len={seq_len}). "
                            f"Consider increasing per_source_loss_sample_interval."
                        )
                    return {}

            with torch.no_grad():
                sampled_batch = {
                    k: v.to(model_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in self._sampled_batch.items()
                }
                # Extract labels BEFORE forward — DO NOT pass labels to model.forward().
                # When fused cross entropy (e.g. LigerFusedLinearCrossEntropyLoss) is active,
                # model.forward(labels=...) computes loss directly from hidden_states x lm_head
                # without materializing the full logits tensor, so outputs.logits will be None.
                # By omitting labels, we force the model to take the explicit path:
                #   logits = self.lm_head(hidden_states)
                labels = sampled_batch.pop("labels")
                # Remove pipeline metadata (if any) before model forward.
                pop_batch_metadata(sampled_batch)
                # Backward compatibility: also pop legacy top-level metadata keys.
                for _key in ("source_boundaries", "ds_idx", "cur_token_num", "source_name"):
                    sampled_batch.pop(_key, None)

                outputs = model(
                    **sampled_batch,
                    use_cache=False,
                )

                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                if logits is None:
                    # Safety fallback: if logits is still None despite omitting labels,
                    # skip gracefully rather than crashing training.
                    if self.global_rank == 0 or self.log_rank == -1:
                        logger.warning(
                            "Per-source loss: model returned logits=None even without labels. "
                            "This may indicate an incompatible model architecture. Skipping."
                        )
                    return {}

                # Group source segments by source_name for efficient computation
                for meta in self._source_metadata:
                    source_name = meta["source_name"]
                    start, end = meta["start_pos"], meta["end_pos"]

                    if end - start <= 1:
                        # Skip segments that are too short for next-token prediction
                        continue

                    # Handle sequence shift (next token prediction)
                    source_logits = logits[:, start : end - 1, :]
                    source_labels = labels[:, start + 1 : end]

                    # Flatten
                    shift_logits = source_logits.view(-1, source_logits.size(-1))
                    shift_labels = source_labels.view(-1)

                    # Filter padding
                    valid_mask = shift_labels != -100
                    if valid_mask.sum() > 0:
                        # Compute loss for this segment
                        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                        segment_loss = loss_fct(shift_logits[valid_mask], shift_labels[valid_mask])

                        # Aggregate by source
                        source_losses[source_name].append(segment_loss.mean().item())

            # Compute average loss per source
            for source_name, losses in source_losses.items():
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    per_source_metrics[f"loss/{source_name}"] = avg_loss

        except RuntimeError as e:
            # GPU OOM or CUDA errors
            if self.global_rank == 0 or self.log_rank == -1:
                logger.warning(
                    f"Per-source loss computation failed (OOM?): {e}. "
                    f"Consider increasing sample_interval or disabling this feature."
                )
        except Exception as e:
            # Other errors - log with context
            if self.global_rank == 0 or self.log_rank == -1:
                logger.error(
                    f"Per-source loss computation failed: {e}",
                    exc_info=True,
                )
        finally:
            # Clear sampled state
            self._sampled_batch = None
            self._source_metadata = []
            self._has_captured = False

        return per_source_metrics

    def should_log(self) -> bool:
        """Check if this rank should log."""
        return self.log_rank == -1 or self.global_rank == self.log_rank


def enable_high_precision_for_bf16():
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if IS_NPU_AVAILABLE:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["NCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if IS_NPU_AVAILABLE:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed_func(seed)


def create_logger(name: Optional[str] = None) -> "logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = builtin_logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(builtin_logging.INFO)
    logger.propagate = False
    return logger


def enable_third_party_logging() -> None:
    """
    Enables explicit logger of the third-party libraries.
    """
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def disable_warning() -> None:
    """
    Enables warning filter.
    """
    from pyiceberg.metrics import LoggingMetricsReporter

    builtin_logging.basicConfig(level=builtin_logging.ERROR)
    warnings.simplefilter("ignore")
    LoggingMetricsReporter()
    LoggingMetricsReporter._logger = builtin_logging.getLogger(LoggingMetricsReporter.__name__)
    LoggingMetricsReporter._logger.setLevel(builtin_logging.WARNING)
    LoggingMetricsReporter._logger.propagate = False


def print_device_mem_info(prompt: str = "VRAM usage") -> None:
    """
    Logs VRAM info.
    """
    memory_allocated = get_torch_device().memory_allocated() / (1024**3)
    max_memory_allocated = get_torch_device().max_memory_allocated() / (1024**3)
    logger.info_rank0(f"{prompt}: cur {memory_allocated:.2f}GB, max {max_memory_allocated:.2f}GB.")


def print_cpu_memory_info():
    cpu_usage = psutil.cpu_percent(interval=1)  # sampling for 1 sec
    logger.info_rank0(f"CPU Usage: {cpu_usage}%")

    memory_info = psutil.virtual_memory()
    logger.info_rank0(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
    logger.info_rank0(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
    logger.info_rank0(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
    logger.info_rank0(f"Memory Usage: {memory_info.percent}%")


def empty_cache() -> None:
    """
    Collects system memory.
    """
    gc.collect()

    if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE:
        from veomni.utils.device import empty_cache

        empty_cache()


def get_cache_dir(path: Optional[str] = None) -> str:
    """
    Returns the cache directory for the given path.
    """
    if path is None:
        return CACHE_DIR

    path = os.path.normpath(path)
    if not os.path.splitext(path)[-1]:  # is a dir
        path = os.path.join(path, "")

    path = os.path.split(os.path.dirname(path))[-1]
    return os.path.join(CACHE_DIR, path, "")  # must endswith os.path.sep


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L350
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


def unwrap_model(model: "nn.Module") -> "nn.Module":
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py#L4808
    """
    if hasattr(model, "module"):
        return unwrap_model(getattr(model, "module"))
    else:
        return model


def print_example(example: Dict[str, "torch.Tensor"], rank: int, print_tensor: bool = True) -> None:
    """
    Logs a single example to screen.
    """
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            if print_tensor:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")
            else:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}")
        else:
            logger.info(f"[rank {rank}]: {key}'s value: {value}")


def dict2device(input_dict: dict):
    """
    Move a dict of Tensor to GPUs.
    """
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.to(get_device_type())
        elif isinstance(v, dict):
            output_dict[k] = dict2device(v)
        else:
            output_dict[k] = v
    return output_dict


def make_list(item):
    if isinstance(item, List) or isinstance(item, np.ndarray):
        return item
    return [item]


class ProfilerWithMem:
    """Thin wrapper that toggles CUDA-allocator tracing around profiler.step()"""

    def __init__(self, inner):
        self._p = inner

    # delegate ctx-manager behaviour
    def __enter__(self):
        return self._p.__enter__()

    def __exit__(self, *a):
        return self._p.__exit__(*a)

    def start(self):
        out = self._p.start()
        get_torch_device().memory._record_memory_history()
        return out

    def stop(self):
        out = self._p.stop()
        get_torch_device().memory._record_memory_history(enabled=None)  # step recording memory snapshot
        return out

    def step(self, *a, **kw):
        return self._p.step(*a, **kw)


def create_profiler(
    start_step: int,
    end_step: int,
    trace_dir: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    global_rank: int,
):
    """
    Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
    Profile steps in [start_step, end_step).

    When is_npu_available = True, the profiler will be created as torch_npu.profiler.

    Args:
        start_step (int): The step to start recording.
        end_step (int): The step to end recording.
        trace_dir (str): The path to save the profiling result.
        record_shapes (bool): Whether to record the shapes of the tensors.
        profile_memory (bool): Whether to profile the memory usage.
        with_stack (bool): Whether to include the stack trace.
    """

    def handler_fn(p):
        time = int(datetime.datetime.now().timestamp())

        trace_file_extention = "pt.trace.json.gz"
        gpu_memory_file_extension = "pkl"

        if trace_dir.startswith("hdfs://"):
            hdfs_io.makedirs(trace_dir, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            trace_file = os.path.join(CACHE_DIR, f"veomni_rank{global_rank}_{time}.{trace_file_extention}")
            gpu_memory_file = os.path.join(CACHE_DIR, f"veomni_rank{global_rank}_{time}.{gpu_memory_file_extension}")
        else:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"veomni_rank{global_rank}_{time}.{trace_file_extention}")
            gpu_memory_file = os.path.join(trace_dir, f"veomni_rank{global_rank}_{time}.{gpu_memory_file_extension}")

        if IS_NPU_AVAILABLE:
            nonlocal npu_trace_handler
            npu_trace_handler(p)
            trace_file = p.prof_if.prof_path
        elif IS_CUDA_AVAILABLE:
            p.export_chrome_trace(trace_file)
        logger.info(f"Profiling result saved at {trace_file}.")

        get_torch_device().memory._dump_snapshot(gpu_memory_file)
        logger.info(f"Profiling memory visualization saved at {gpu_memory_file}.")

        if trace_dir.startswith("hdfs://"):
            copy(trace_file, trace_dir)
            logger.info(f"Profiling result uploaded to {trace_dir}.")

        if VEOMNI_UPLOAD_CMD:
            try:
                logger.info_rank0(f"upload trace file {trace_file}")
                if IS_NPU_AVAILABLE:
                    import gzip
                    import shutil

                    npu_trace_file = f"{trace_file}/ASCEND_PROFILER_OUTPUT/trace_view.json"
                    with open(npu_trace_file, "rb") as f_in, gzip.open(f"{npu_trace_file}.gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    command2 = f"{VEOMNI_UPLOAD_CMD} {npu_trace_file}.gz"
                else:
                    command2 = f"{VEOMNI_UPLOAD_CMD} {trace_file}"
                subprocess.run(command2, shell=True, check=True, executable="/bin/bash")
            except Exception as e:
                logger.warning(f"failed to upload trace file {trace_file}, error: {e}")

    if IS_NPU_AVAILABLE:
        profiler_module = torch_npu.profiler
        activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.NPU]
        npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
            CACHE_DIR if trace_dir.startswith("hdfs://") else trace_dir
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
        )
    else:
        profiler_module = torch.profiler
        activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.CUDA]
        experimental_config = None

    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")

    schedule = profiler_module.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=1,
    )
    base_profiler = profiler_module.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
        experimental_config=experimental_config,
    )
    if IS_CUDA_AVAILABLE and profile_memory:
        return ProfilerWithMem(base_profiler)
    else:
        return base_profiler


if os.getenv("DISABLE_WARNINGS", "0").lower() in ["true", "1"]:
    disable_warning()
