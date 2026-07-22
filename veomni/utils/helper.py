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
import multiprocessing
import os
import random
import shlex
import subprocess
import sys
import time
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
# Platform integrations assign this module global after import. An explicit
# user override is read separately from the environment in ``handler_fn``.
VEOMNI_UPLOAD_CMD = None
FlopsCounter = None

# Offline Ascend postprocess sidecar (analyse / durable copy / upload).
# - unset / auto: upload on Merlin when a JobRun context is available
# - VEOMNI_NPU_OFFLINE_POSTPROCESS=1: always spawn after raw finalize
# - VEOMNI_NPU_OFFLINE_POSTPROCESS=0: never spawn; preserve the local raw capture only
VEOMNI_NPU_OFFLINE_POSTPROCESS = os.getenv("VEOMNI_NPU_OFFLINE_POSTPROCESS")
VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD = os.getenv("VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD")


def _env_flag(value: Optional[str]) -> Optional[bool]:
    if value is None or value == "":
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def validate_npu_profile_config(trace_dir: str, npu_analysis_mode: str) -> None:
    """Validate rank-shared NPU options before rank-local profiler creation."""
    if IS_NPU_AVAILABLE and npu_analysis_mode == "async" and trace_dir.startswith("hdfs://"):
        raise ValueError(
            "NPU async analysis requires a pod-local trace_dir. Background analysis is still writing its output, "
            "so an hdfs:// destination cannot be copied safely from on_trace_ready."
        )


def _should_upload_npu_profile_to_merlin() -> bool:
    """Auto-enable Merlin upload in a JobRun; the sidecar selects the uploader."""
    configured = _env_flag(VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD)
    if configured is False:
        return False

    has_merlin_context = bool(
        os.getenv("RH2_JOB_RUN_ID") or os.getenv("MERLIN_JOB_ID") or os.getenv("ARNOLD_TRIAL_ID")
    )
    return configured is True or has_merlin_context


def spawn_npu_offline_sidecar(
    raw_dir: str,
    *,
    copy_to: Optional[str] = None,
    analyse: bool = True,
    upload_cmd: Optional[str] = None,
    merlin_upload: bool = False,
    platform_associated_upload: bool = False,
    job_associated_upload: bool = False,
) -> Optional[subprocess.Popen]:
    """Fire-and-forget offline Ascend postprocess so training is not blocked.

    The sidecar runs in a new session so it can outlive a soft train shutdown.
    Logs go to ``<raw_dir>/veomni_npu_offline_postprocess.log``.
    """
    cmd = [sys.executable, "-m", "veomni.utils.npu_offline_postprocess", "--raw-dir", raw_dir]
    if copy_to:
        cmd.extend(["--copy-to", copy_to])
    if analyse:
        cmd.append("--analyse")
    if upload_cmd:
        cmd.extend(["--upload-cmd", upload_cmd])
    if merlin_upload:
        cmd.append("--merlin-upload")

    if not (copy_to or analyse or upload_cmd or merlin_upload):
        logger.warning("spawn_npu_offline_sidecar called with nothing to do; skipping")
        return None

    log_path = os.path.join(
        raw_dir if os.path.isdir(raw_dir) else os.path.dirname(raw_dir) or ".",
        "veomni_npu_offline_postprocess.log",
    )
    log_fh = None
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8")
        sidecar_env = None
        if platform_associated_upload:
            trial_id = os.getenv("ARNOLD_TRIAL_ID")
            job_id = os.getenv("RH2_JOB_RUN_ID") or os.getenv("MERLIN_JOB_ID")
            if trial_id:
                sidecar_env = os.environ.copy()
                sidecar_env["ARNOLD_TRIAL_ID"] = trial_id
                sidecar_env.pop("RH2_JOB_RUN_ID", None)
                sidecar_env.pop("MERLIN_JOB_ID", None)
                sidecar_env.pop("ARNOLD_RUN_ID", None)
            elif job_id:
                sidecar_env = os.environ.copy()
                sidecar_env["MERLIN_JOB_ID"] = job_id
                sidecar_env.pop("ARNOLD_TRIAL_ID", None)
                sidecar_env.pop("ARNOLD_RUN_ID", None)
        elif job_associated_upload:
            # Compatibility for direct callers of the former keyword: retain
            # its JobRun-first environment semantics.
            job_id = os.getenv("RH2_JOB_RUN_ID") or os.getenv("MERLIN_JOB_ID")
            if job_id:
                sidecar_env = os.environ.copy()
                sidecar_env["MERLIN_JOB_ID"] = job_id
                sidecar_env.pop("ARNOLD_TRIAL_ID", None)
                sidecar_env.pop("ARNOLD_RUN_ID", None)
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=sidecar_env,
        )
    except Exception as exc:
        logger.warning(f"Failed to spawn NPU offline postprocess sidecar: {exc}")
        return None
    finally:
        if log_fh is not None:
            log_fh.close()

    logger.info(f"Spawned NPU offline postprocess sidecar pid={proc.pid} log={log_path} cmd={shlex.join(cmd)}")
    return proc


def wait_npu_profile_sidecars(profiler, timeout_seconds: float = 300.0) -> None:
    """Wait briefly for detached NPU postprocess work after training is done."""
    sidecars = getattr(profiler, "_veomni_npu_sidecars", ())
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    for proc in sidecars:
        started = time.perf_counter()
        try:
            return_code = proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired:
            logger.warning(
                f"NPU profile sidecar pid={proc.pid} is still running after {timeout_seconds:.1f}s; "
                "the raw local capture is preserved, but durable copy/upload may still be incomplete."
            )
            continue
        duration = time.perf_counter() - started
        if return_code == 0:
            logger.info(f"NPU_PROFILE_SIDECAR_WAIT pid={proc.pid} status=completed duration_seconds={duration:.6f}")
        else:
            logger.warning(
                f"NPU profile sidecar pid={proc.pid} exited with return code {return_code}; "
                "inspect veomni_npu_offline_postprocess.log and preserve the raw capture."
            )


def convert_hdfs_fuse_path(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    return kwargs.get("path", None)


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PretrainedConfig

    from ..distributed.parallel_state import ParallelState


logger = logging.get_logger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))


def _compute_seqlens(micro_batch: Dict[str, "torch.Tensor"]) -> List[int]:
    if "cu_seq_lens_q" in micro_batch:
        # packed micro batch
        tail_padding_length = micro_batch.get("tail_padding_length")
        seqlens = valid_seqlens_from_cu_seqlens(
            micro_batch["cu_seq_lens_q"],
            tail_padding_length=int(tail_padding_length) if tail_padding_length is not None else None,
        ).tolist()
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
    if "latents" not in micro_batch:
        return []
    dit_latents_seqlens = []
    for latents in micro_batch["latents"]:
        latent_shape = latents.shape
        if len(latent_shape) == 3:
            B, seq_len, _C = latent_shape
            dit_latents_seqlens.append(B * seq_len)
            continue
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
    ds_idx = micro_batch.pop("ds_idx")
    micro_batch.pop("source_name", None)
    micro_batch.pop("cur_token_num", None)
    if isinstance(ds_idx, torch.Tensor):
        # packed micro batch
        return ds_idx.tolist()
    else:
        # unpacked sample
        return [ds_idx]


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
        parallel_state: Optional["ParallelState"] = None,
    ) -> None:
        self.config = config
        self.global_batch_size = global_batch_size
        self.enable_multisource = enable_multisource
        self.empty_cache_steps = empty_cache_steps
        self.gc_steps = gc_steps

        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
        self.world_size = dist.get_world_size()
        self.consume_tokens = 0
        self.consume_chunks = 0
        self.batch_seqlens = []
        self.batch_ds_idx = []
        self.images_seqlens = []

        if self.enable_multisource:
            if dataloader is None or data_path is None:
                raise ValueError(
                    "`dataloader` and `data_path` is required for `EnvironMeter` with multi-source dataloader."
                )

            self.multisource_tracker = MultiSourceInfoTracker(
                dataloader=dataloader, data_path=data_path, parallel_state=self.parallel_state
            )

        # for internal use
        if VALID_CONFIG_TYPE is not None and isinstance(config, VALID_CONFIG_TYPE):
            self.estimate_flops = FlopsCounter(config).estimate_flops
        else:
            self.estimate_flops = VeomniFlopsCounter(config).estimate_flops

        if self.gc_steps > 0:
            gc.disable()

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {"consume_tokens": self.consume_tokens, "consume_chunks": self.consume_chunks}
        if self.enable_multisource:
            state_dict.update({"multisource_tracker": self.multisource_tracker.state_dict()})

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.consume_tokens = state_dict["consume_tokens"]
        self.consume_chunks = state_dict["consume_chunks"]
        if self.enable_multisource:
            self.multisource_tracker.load_state_dict(state_dict["multisource_tracker"])

    def add(self, micro_batch: Union[Dict[str, "torch.Tensor"], List[Dict[str, "torch.Tensor"]]]) -> None:
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

    def step(self, delta_time: float, global_step: int) -> Dict[str, Any]:
        if len(self.images_seqlens) > 0:
            flops_achieved, flops_promised = self.estimate_flops(
                self.batch_seqlens, delta_time, images_seqlens=self.images_seqlens
            )
        else:
            flops_achieved, flops_promised = self.estimate_flops(self.batch_seqlens, delta_time)
        flops_achieved, batch_tokens, real_global_batch_size = all_reduce(
            (flops_achieved, sum(self.batch_seqlens), len(self.batch_seqlens)),
            op="sum",
            group=self.parallel_state.dp_group,
        )
        flops_promised = flops_promised * self.world_size
        mfu = flops_achieved / flops_promised if flops_promised else 0

        # calculate average effective len and tokens per second
        avg_effective_len = batch_tokens / self.global_batch_size if self.global_batch_size else 0
        avg_sample_seq_len = batch_tokens / real_global_batch_size if real_global_batch_size else 0
        tokens_per_second = batch_tokens / delta_time
        self.consume_tokens += batch_tokens
        self.consume_chunks += real_global_batch_size

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
            "max_memory_allocated(GB)": allocated_memory / (1024**3),
            "max_memory_reserved(GB)": reserved_memory / (1024**3),
            "cpu_used_memory(GB)": cpu_memory_info.used / (1024**3),
            "cpu_available_memory(GB)": cpu_memory_info.available / (1024**3),
            "cpu_memory_usage(%)": cpu_memory_info.percent,
            "num_alloc_retries": num_alloc_retries,
        }

        if self.enable_multisource:
            metrics.update(self.multisource_tracker.step(self.batch_ds_idx, self.batch_seqlens))

        if self.empty_cache_steps > 0 and global_step % self.empty_cache_steps == 0:
            empty_cache()

        if self.gc_steps > 0 and global_step % self.gc_steps == 0:
            gc.collect()

        self.batch_seqlens = []
        self.batch_ds_idx = []
        self.images_seqlens = []

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
    Tracks the statistics about the weighted multi-source dataset.
    """

    def __init__(
        self,
        dataloader: Optional["DataLoader"],
        data_path: str,
        parallel_state: Optional["ParallelState"] = None,
    ) -> None:
        self.dataloader = dataloader
        self.parallel_state = parallel_state if parallel_state is not None else get_parallel_state()
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
        Computes the statistics about the weighted multi-source dataset. It should be called at every rank to update dataloader.
        """
        counter = defaultdict(MultiSourceCounterItem)
        for ds_idx, seq_len in zip(batch_ds_idx, batch_seqlens):
            counter[ds_idx].increment(seq_len, 1)

        counter_list: List[Dict[int, MultiSourceCounterItem]] = [None for _ in range(self.parallel_state.dp_size)]
        dist.all_gather_object(counter_list, counter, group=self.parallel_state.dp_group)

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
            not self.parallel_state.tp_enabled or self.parallel_state.tp_rank == 0
        ):  # update at every dp rank
            if self.boundary_type == "token":
                self.dataloader.update_consumed_tokens((self.batch_idx, global_consumed_tokens))
            elif self.boundary_type == "sample":
                self.dataloader.update_consumed_tokens((self.batch_idx, global_comsumed_samples))

        self.batch_idx += 1
        multisource_info = {}
        for ds_idx, _item in self.accumulate_counter.items():
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
    if get_device_type() == "cpu":
        print_cpu_memory_info()
    else:
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
        return unwrap_model(model.module)
    else:
        return model


def print_example(example: Dict[str, "torch.Tensor"], rank: int, print_tensor: bool = True) -> None:
    """
    Logs a single example to screen.

    Nested dicts (e.g. ``multimodal_metadata`` from ``PackingCollator``)
    are expanded one level so inner tensor shapes/devices stay visible
    instead of being collapsed into a single dict-repr line.
    """

    def _log(key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if print_tensor:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")
            else:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}")
        else:
            logger.info(f"[rank {rank}]: {key}'s value: {value}")

    for key, value in example.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                _log(f"{key}[{inner_key!r}]", inner_value)
        else:
            _log(key, value)


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
    with_modules: bool,
    global_rank: int,
    npu_analysis_mode: str = "offline",
    npu_offline_analysis: Optional[bool] = None,
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
        npu_analysis_mode (str): Ascend analysis mode: ``offline`` or ``async``.
        npu_offline_analysis (bool, optional): Deprecated alias; ``True`` maps to offline.
    """

    if npu_offline_analysis is True:
        if npu_analysis_mode == "async":
            raise ValueError(
                "Conflicting NPU profiler options: npu_offline_analysis=True and npu_analysis_mode='async'."
            )
        warnings.warn(
            "npu_offline_analysis is deprecated; use npu_analysis_mode='offline'.",
            DeprecationWarning,
            stacklevel=2,
        )
        npu_analysis_mode = "offline"
    elif npu_offline_analysis is False:
        raise ValueError(
            "npu_offline_analysis=False requested removed synchronous online analysis; choose "
            "npu_analysis_mode='async' or npu_analysis_mode='offline' explicitly."
        )

    if npu_analysis_mode not in {"offline", "async"}:
        raise ValueError(f"Invalid npu_analysis_mode={npu_analysis_mode!r}; expected one of: offline, async.")

    validate_npu_profile_config(trace_dir, npu_analysis_mode)
    is_hdfs_trace = trace_dir.startswith("hdfs://")

    effective_npu_analysis_mode = npu_analysis_mode
    npu_sidecars: list[subprocess.Popen] = []

    def handler_fn(p):
        timestamp = int(datetime.datetime.now().timestamp())

        trace_file_extention = "pt.trace.json.gz"
        gpu_memory_file_extension = "pkl"

        if is_hdfs_trace:
            if not IS_NPU_AVAILABLE:
                hdfs_io.makedirs(trace_dir, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            trace_file = os.path.join(CACHE_DIR, f"veomni_rank{global_rank}_{timestamp}.{trace_file_extention}")
            gpu_memory_file = os.path.join(
                CACHE_DIR, f"veomni_rank{global_rank}_{timestamp}.{gpu_memory_file_extension}"
            )
        else:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"veomni_rank{global_rank}_{timestamp}.{trace_file_extention}")
            gpu_memory_file = os.path.join(
                trace_dir, f"veomni_rank{global_rank}_{timestamp}.{gpu_memory_file_extension}"
            )

        if IS_NPU_AVAILABLE:
            nonlocal npu_trace_handler
            trace_file = p.prof_if.prof_path
            handler_started = time.perf_counter()
            handler_status = "ok"
            try:
                npu_trace_handler(p)
            except Exception as exc:
                if effective_npu_analysis_mode != "async":
                    raise
                # Raw finalization has already selected prof_path. A failure to
                # submit optional background analysis must not take down the
                # distributed training job or strand peers at the barrier.
                handler_status = "analysis_submit_failed"
                logger.warning(
                    "NPU async analysis submission failed; training will continue and the finalized raw capture "
                    f"remains at {trace_file}. Error: {exc}"
                )
            handler_seconds = time.perf_counter() - handler_started
            logger.info(
                "NPU_PROFILE_HANDLER "
                f"mode={effective_npu_analysis_mode} status={handler_status} rank={global_rank} "
                f"duration_seconds={handler_seconds:.6f} raw_dir={trace_file}"
            )
        elif IS_CUDA_AVAILABLE:
            p.export_chrome_trace(trace_file)
        logger.info(f"Profiling result saved at {trace_file}.")

        if IS_NPU_AVAILABLE:
            offline_postprocess_flag = _env_flag(VEOMNI_NPU_OFFLINE_POSTPROCESS)
            merlin_upload = _should_upload_npu_profile_to_merlin()
            user_upload_cmd = os.getenv("VEOMNI_UPLOAD_CMD")
            platform_upload_cmd = VEOMNI_UPLOAD_CMD
            automatic_upload_opted_out = _env_flag(VEOMNI_NPU_OFFLINE_MERLIN_UPLOAD) is False
            if user_upload_cmd:
                selected_upload_cmd = user_upload_cmd
                selected_merlin_upload = False
                selected_platform_associated_upload = False
            elif platform_upload_cmd and not automatic_upload_opted_out:
                # Platform integrations may provide a file-based uploader that
                # handles traces too large for an SDK JSON/base64 request. Give
                # it a trial-first environment because the JobRun Profiling
                # tab queries assets through the selected Arnold trial.
                selected_upload_cmd = platform_upload_cmd
                selected_merlin_upload = False
                selected_platform_associated_upload = merlin_upload
            elif merlin_upload:
                selected_upload_cmd = None
                selected_merlin_upload = True
                selected_platform_associated_upload = False
            else:
                selected_upload_cmd = None
                selected_merlin_upload = False
                selected_platform_associated_upload = False

            if effective_npu_analysis_mode == "async":
                if offline_postprocess_flag is True or selected_upload_cmd or selected_merlin_upload:
                    logger.warning(
                        "Automatic copy/upload is skipped for NPU async analysis because the background parser may "
                        f"still be writing {trace_file}. Upload the completed trace after training exits."
                    )
                return

            needs_copy = is_hdfs_trace
            needs_analysis = offline_postprocess_flag is True or bool(selected_upload_cmd) or selected_merlin_upload
            needs_sidecar = needs_copy or needs_analysis
            if needs_sidecar and offline_postprocess_flag is not False:
                sidecar_kwargs = {
                    "copy_to": trace_dir if needs_copy else None,
                    "analyse": needs_analysis,
                    "upload_cmd": selected_upload_cmd,
                    "merlin_upload": selected_merlin_upload,
                }
                if selected_platform_associated_upload:
                    sidecar_kwargs["platform_associated_upload"] = True
                proc = spawn_npu_offline_sidecar(str(trace_file), **sidecar_kwargs)
                if proc is None:
                    logger.warning(
                        "NPU offline postprocess sidecar did not start; no synchronous fallback will run inside the "
                        f"training barrier. The raw capture remains at {trace_file}."
                    )
                else:
                    npu_sidecars.append(proc)
            elif needs_sidecar:
                logger.warning(
                    "NPU offline postprocess sidecar is disabled; no synchronous copy, analysis, or upload will run "
                    f"inside the training barrier. The raw capture remains at {trace_file}."
                )
            return

        get_torch_device().memory._dump_snapshot(gpu_memory_file)
        logger.info(f"Profiling memory visualization saved at {gpu_memory_file}.")

        if is_hdfs_trace:
            copy(trace_file, trace_dir)
            logger.info(f"Profiling result uploaded to {trace_dir}.")

        if VEOMNI_UPLOAD_CMD:
            try:
                logger.info_rank0(f"upload trace file {trace_file}")
                command2 = f"{VEOMNI_UPLOAD_CMD} {trace_file}"
                subprocess.run(command2, shell=True, check=True, executable="/bin/bash")
            except Exception as e:
                logger.warning(f"failed to upload trace file {trace_file}, error: {e}")

    if IS_NPU_AVAILABLE:
        profiler_module = torch_npu.profiler
        activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.NPU]
        npu_trace_dir = CACHE_DIR if is_hdfs_trace else trace_dir
        if npu_analysis_mode == "async" and multiprocessing.current_process().daemon:
            effective_npu_analysis_mode = "offline"
            logger.warning(
                "NPU async analysis is unavailable in a daemon process; falling back to offline raw capture."
            )

        if effective_npu_analysis_mode == "async":
            try:
                npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
                    npu_trace_dir,
                    analyse_flag=True,
                    async_mode=True,
                )
            except TypeError as exc:
                effective_npu_analysis_mode = "offline"
                logger.warning(
                    "This torch_npu version does not support async tensorboard trace analysis; falling back to "
                    f"offline raw capture. Error: {exc}"
                )
                npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
                    npu_trace_dir,
                    analyse_flag=False,
                )
        else:
            npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
                npu_trace_dir,
                analyse_flag=False,
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
        with_modules=with_modules,
        with_stack=with_stack,
        experimental_config=experimental_config,
    )
    if IS_NPU_AVAILABLE:
        base_profiler._veomni_npu_analysis_mode = effective_npu_analysis_mode
        base_profiler._veomni_npu_sidecars = npu_sidecars
        return base_profiler
    if IS_CUDA_AVAILABLE and profile_memory:
        return ProfilerWithMem(base_profiler)
    return base_profiler


if os.getenv("DISABLE_WARNINGS", "0").lower() in ["true", "1"]:
    disable_warning()
