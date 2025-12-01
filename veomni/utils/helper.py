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
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers import enable_full_determinism
from transformers import set_seed as set_seed_func

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils import logging
from veomni.utils.count_flops import VeomniFlopsCounter
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_torch_device,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.seqlen_pos_transform_utils import culen2len, pos2culen

from .import_utils import is_veomni_patch_available
from .multisource_utils import parse_multisource_config


try:
    import hdfs_io
    from hdfs_io import copy
except (ImportError, ModuleNotFoundError):
    from veomni.utils import hdfs_io
    from veomni.utils.hdfs_io import copy

if IS_NPU_AVAILABLE:
    import torch_npu


if is_veomni_patch_available():
    from veomni_patch.utils.helper import (
        VALID_CONFIG_TYPE,
        VEOMNI_UPLOAD_CMD,
        FlopsCounter,
        convert_hdfs_fuse_path,
        is_remote_path,
        load_step2token,
        save_step2token,
    )
else:

    def load_step2token(*args, **kwargs):
        logger.warning("veomni_patch is not available, load_step2token will be skipped")
        pass

    def save_step2token(*args, **kwargs):
        logger.warning("veomni_patch is not available, save_step2token will be skipped")
        pass

    def is_remote_path(*args, **kwargs):
        logger.warning("veomni_patch is not available, is_remote_path returning False")
        return False

    def convert_hdfs_fuse_path(*args, **kwargs):
        logger.warning("veomni_patch is not available, convert_hdfs_fuse_path returning path as-is")
        if len(args) > 0:
            return args[0]
        return kwargs.get("path", None)

    VALID_CONFIG_TYPE = None
    VEOMNI_UPLOAD_CMD = None

    class FlopsCounter:
        def __init__(self):
            raise ImportError("veomni_patch is not available, please install it first")


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PretrainedConfig


logger = logging.get_logger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))


def _compute_seqlens(
    micro_batch: Dict[str, "torch.Tensor"], rmpad: bool, rmpad_with_pos_ids: bool, enable_multisource: bool
) -> Tuple[List[int], Optional[List[int]]]:
    """
    Computes the sequence lengths of the current batch.

    Args:
        micro_batch (Dict[str, Tensor]): The current batch.
        rmpad (bool): Whether to remove the padding tokens.
        rmpad_with_pos_ids (bool): Whether to remove the padding tokens using the position ids.
        enable_multisource (bool): Whether to enable the multi-source dataloader.
    """
    attention_mask = micro_batch["attention_mask"]
    if rmpad:
        seqlens = culen2len(micro_batch["cu_seqlens"]).tolist()
        seqlens = seqlens[:-1] if (attention_mask == 0).any().item() else seqlens
    elif rmpad_with_pos_ids:
        seqlens = culen2len(pos2culen(micro_batch["position_ids"])).tolist()
        seqlens = seqlens[:-1] if (attention_mask == 0).any().item() else seqlens
    else:
        seqlens = attention_mask.sum(-1).tolist()

    ds_idx = None
    if enable_multisource:
        ds_idx = micro_batch["ds_idx"].tolist()

    return seqlens, ds_idx


class EnvironMeter:
    """
    Computes the metrics about the training efficiency.

    Args:
        config (PretrainedConfig): The configuration of the model.
        global_batch_size (int): The global batch size.
        rmpad (bool, optional): Whether to remove the padding tokens. Defaults to False.
        rmpad_with_pos_ids (bool, optional): Whether to remove the padding tokens using the position ids. Defaults to False.
        enable_multisource (bool, optional): Whether to enable the multi-source dataloader. Defaults to False.
        dataloader (DataLoader, optional): The training dataloader for multi-source dataloader. Defaults to None.
        data_path (str, optional): The data path for multi-source dataloader. Defaults to "".
        empty_cache_steps (int, optional): The number of steps to empty the cache. Defaults to 500.
    """

    def __init__(
        self,
        config: "PretrainedConfig",
        global_batch_size: int,
        rmpad: bool = False,
        rmpad_with_pos_ids: bool = False,
        enable_multisource: bool = False,
        dataloader: Optional["DataLoader"] = None,
        data_path: str = "",
        empty_cache_steps: int = 500,
        gc_steps: int = 0,
    ) -> None:
        self.config = config
        self.global_batch_size = global_batch_size
        self.rmpad = rmpad
        self.rmpad_with_pos_ids = rmpad_with_pos_ids
        self.enable_multisource = enable_multisource
        self.empty_cache_steps = empty_cache_steps
        self.gc_steps = gc_steps
        self.world_size = dist.get_world_size()
        self.consume_tokens = 0
        self.batch_seqlens = []
        self.batch_ds_idx = []
        self.image_seqlens = []

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
        state_dict = {"consume_tokens": self.consume_tokens}
        if self.enable_multisource:
            state_dict.update({"multisource_tracker": self.multisource_tracker.state_dict()})

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.consume_tokens = state_dict["consume_tokens"]
        if self.enable_multisource:
            self.multisource_tracker.load_state_dict(state_dict["multisource_tracker"])

    def add(self, micro_batch: Dict[str, "torch.Tensor"]) -> None:
        seqlens, ds_idx = _compute_seqlens(micro_batch, self.rmpad, self.rmpad_with_pos_ids, self.enable_multisource)

        if "image_grid_thw" in micro_batch:
            image_grid_thw = micro_batch["image_grid_thw"]
            image_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            self.image_seqlens.extend(image_seqlens.tolist())

        if "video_grid_thw" in micro_batch:
            video_grid_thw = micro_batch["video_grid_thw"]
            video_seqlens = torch.repeat_interleave(video_grid_thw[:, 1] * video_grid_thw[:, 2], video_grid_thw[:, 0])
            self.image_seqlens.extend(video_seqlens.tolist())  # video equals to image

        if self.enable_multisource:
            self.batch_seqlens.extend(seqlens[: len(ds_idx)])  # rmpad_with_pos_ids has a pad item
            self.batch_ds_idx.extend(ds_idx)
        else:
            self.batch_seqlens.extend(seqlens)

    def step(self, delta_time: float, global_step: int) -> Dict[str, Any]:
        if len(self.image_seqlens) > 0:
            flops_achieved, flops_promised = self.estimate_flops(
                self.batch_seqlens, delta_time, image_seqlens=self.image_seqlens
            )
        else:
            flops_achieved, flops_promised = self.estimate_flops(self.batch_seqlens, delta_time)

        flops_achieved, batch_tokens, real_global_batch_size = all_reduce(
            (flops_achieved, sum(self.batch_seqlens), len(self.batch_seqlens)),
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
        self.image_seqlens = []

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


def set_seed(seed: int, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if full_determinism:
        enable_full_determinism(seed, warn_only=True)
    else:
        set_seed_func(seed)


def create_logger(name: Optional[str] = None) -> "logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
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
    cpu_usage = psutil.cpu_percent(interval=1)  # 1 秒间隔
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
        self.first_step = True  # flagging the first step for record memory history

    # delegate ctx-manager behaviour
    def __enter__(self):
        return self._p.__enter__()

    def __exit__(self, *a):
        return self._p.__exit__(*a)

    def start(self):
        return self._p.start()

    def stop(self):
        out = self._p.stop()
        get_torch_device().memory._record_memory_history(enabled=None)  # step recording memory snapshot
        return out

    def step(self, *a, **kw):
        out = self._p.step(*a, **kw)
        if self.first_step:
            get_torch_device().memory._record_memory_history()
            self.first_step = False
        return out


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

        if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE:
            get_torch_device().memory._dump_snapshot(gpu_memory_file)
            logger.info(f"Profiling memory visualization saved at {gpu_memory_file}.")

        if trace_dir.startswith("hdfs://"):
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
