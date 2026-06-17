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

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .data_loader import DATALOADER_REGISTRY


logger = logging.get_logger(__name__)

BYTED_LOADER_TYPE = "byted_loader"
_TEST_CLIENT_FACTORY: Optional[Callable[..., Any]] = None
_AUDIT_FILE = None

_LOADER_METADATA_KEYS = {
    "source_name",
    "data_name",
    "dataname",
    "ds_idx",
    "source_id",
    "sample_id",
    "epoch",
}


@dataclass
class BytedLoaderDatasetSpec:
    """Dataset placeholder consumed by the bytedance.dataloader backend.

    The byted loader backend owns storage reads, scheduling, shuffling and
    progress state. VeOmni keeps only the transform and source metadata needed
    to bridge into its collator and trainer loop.
    """

    train_path: str
    raw_train_path: str
    source_config_path: str
    transform: Callable[[Mapping[str, Any]], Any]
    is_multisource_yaml: bool
    dataset_name_before_bypass: str
    data_type: str
    shuffle: bool
    shuffle_seed: Optional[int]
    shuffle_shard_nums: int
    ckpt_dir: str
    save_ckpt_interval: int
    disable_veomni_multisource_meter: bool = True


class _DefaultTransformMapping(dict):
    def __init__(self, default_name: str, transforms: List[Callable[[Any], Any]]) -> None:
        super().__init__({default_name: transforms, "default": transforms, "text_transform": transforms})
        self.default_name = default_name

    def __missing__(self, key: str) -> List[Callable[[Any], Any]]:
        value = self[self.default_name]
        self[key] = value
        return value


class VeOmniSampleTransform:
    """Run the VeOmni sample transform inside byted loader workers."""

    def __init__(self, transform: Callable[[Mapping[str, Any]], Any]) -> None:
        self.transform = transform
        self._accepts_source_name = _callable_accepts_keyword(transform, "source_name")

    def __call__(self, sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
        source_name = sample.get("source_name") or sample.get("data_name") or sample.get("dataname")
        if self._accepts_source_name:
            transformed = self.transform(sample, source_name=source_name)
        else:
            transformed = self.transform(sample)
        if transformed is None:
            return []
        if not isinstance(transformed, list):
            transformed = [transformed]
        return [_sanitize_sample(item) for item in transformed if item is not None]


class SanitizeThenCollate:
    """Drop loader metadata before VeOmni MainCollator and model forward."""

    def __init__(self, collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]]) -> None:
        self.collate_fn = collate_fn

    def __call__(self, samples: List[Mapping[str, Any]]) -> Dict[str, Any]:
        return self.collate_fn([_sanitize_sample(sample) for sample in samples])


class BytedLoaderAdapter:
    """Compatibility wrapper for VeOmni's dataloader protocol."""

    def __init__(
        self,
        client: Any,
        *,
        identity: str,
        ckpt_dir: str,
        total_steps: int,
        enable_batch_db_save: bool,
        config_summary: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.identity = identity
        self.ckpt_dir = ckpt_dir
        self.total_steps = total_steps
        self.enable_batch_db_save = enable_batch_db_save
        self.config_summary = dict(config_summary)
        self._epoch = 0
        self._closed = False
        self._iterator = None
        self._yield_index = 0
        self._audit_enabled = _audit_enabled()
        self._audit_limit = _audit_limit()

    def __iter__(self):
        self._iterator = iter(self.client)
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.client)
        if self._audit_enabled:
            started = time.perf_counter()
        else:
            started = 0.0
        batch = next(self._iterator)
        self._yield_index += 1
        if self._audit_enabled:
            _log_adapter_yield(
                identity=self.identity,
                yield_index=self._yield_index,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                batch=batch,
                config_summary=self.config_summary,
                audit_limit=self._audit_limit,
            )
        return batch

    def __len__(self) -> int:
        if hasattr(self.client, "__len__"):
            return len(self.client)
        return self.total_steps

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        client_state = self.client.state_dict() if hasattr(self.client, "state_dict") else {}
        return {
            "client_state": client_state,
            "identity": self.identity,
            "ckpt_dir": self.ckpt_dir,
            "total_steps": self.total_steps,
            "enable_batch_db_save": self.enable_batch_db_save,
            "config_summary": self.config_summary,
            "adapter_epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        client_state = state_dict.get("client_state", state_dict)
        if hasattr(self.client, "load_state_dict"):
            self.client.load_state_dict(client_state)
        self._epoch = int(state_dict.get("adapter_epoch", self._epoch))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if _is_byted_local_mode_client(self.client):
            _force_close_byted_local_mode_client(self.client, self.identity)
            return
        if hasattr(self.client, "close"):
            self.client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def is_byted_loader_type(dataloader_cfg: Any) -> bool:
    return getattr(dataloader_cfg, "type", None) == BYTED_LOADER_TYPE


def set_byted_loader_client_factory_for_tests(factory: Optional[Callable[..., Any]]) -> None:
    global _TEST_CLIENT_FACTORY
    _TEST_CLIENT_FACTORY = factory


def build_byted_loader_dataset_spec(
    args: Any, transform: Callable[[Mapping[str, Any]], Any]
) -> BytedLoaderDatasetSpec:
    _disable_veomni_multisource_meter(args)
    ckpt_dir = _resolve_byted_ckpt_dir(args)
    save_ckpt_interval = _resolve_save_ckpt_interval(args)
    _guard_no_forbidden_write_root(ckpt_dir, "byted loader ckpt_dir")

    return BytedLoaderDatasetSpec(
        train_path=args.data.train_path,
        raw_train_path=args.data.train_path,
        source_config_path=args.data.train_path if str(args.data.train_path).endswith(".yaml") else "",
        transform=transform,
        is_multisource_yaml=str(args.data.train_path).endswith(".yaml"),
        dataset_name_before_bypass=args.data.dataset_name,
        data_type=args.data.data_type,
        shuffle=bool(getattr(args.data, "shuffle", True)),
        shuffle_seed=getattr(args.data, "shuffle_seed", None),
        shuffle_shard_nums=int(getattr(args.data, "shuffle_shard_nums", 1)),
        ckpt_dir=ckpt_dir,
        save_ckpt_interval=save_ckpt_interval,
    )


@DATALOADER_REGISTRY.register(BYTED_LOADER_TYPE)
def build_byted_loader_dataloader(
    dataset: BytedLoaderDatasetSpec,
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    dyn_bsz: bool,
    dyn_bsz_buffer_size: int,
    bsz_warmup_ratio: float,
    seed: int,
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    bsz_warmup_init_mbtoken: int = 0,
    mode: str = "local",
    file_type: str = "lance",
    shuffle: Optional[bool] = None,
    shuffle_seed: Optional[int] = None,
    shuffle_algo: str = "file",
    shuffle_shard_nums: int = 10,
    worker_num: int = -1,
    worker_subprocess_num: int = 4,
    worker_parallel_read_num: int = 4,
    worker_prefetch_num: int = 1_000_000,
    client_prefetch_num: int = 2,
    server_prefetch_num: int = 8,
    ckpt_dir: str = "",
    save_ckpt_interval: int = -1,
    enable_ckpt: bool = True,
    enable_batch_db_save: bool = False,
    resume_ckpt_path: str = "",
    resume_use_latest_snapshot: bool = False,
    allow_transform_failure: bool = False,
    do_sp_split_in_loader: bool = False,
    gpu_prefetch: bool = False,
    pin_memory: bool = True,
    identity: str = "veomni",
    start_role_after_iter: bool = False,
    peek_num: int = 300,
    enable_balance: bool = False,
    microbatch_balance: bool = False,
    biwise_balance: bool = True,
    seq_len_warmup: bool = False,
    num_warmup_steps: int = 0,
    dsp_rotate_interval: int = -1,
    dp_constructor_per_dp: int = 1,
    min_version: str = "0.1.42",
    strict_api_check: bool = True,
    **unused_dataloader_kwargs: Any,
) -> BytedLoaderAdapter:
    del dataloader_batch_size, dyn_bsz, dyn_bsz_buffer_size, bsz_warmup_ratio

    if unused_dataloader_kwargs:
        logger.warning_rank0(f"[byted_loader] ignoring native dataloader kwargs: {sorted(unused_dataloader_kwargs)}")

    if not isinstance(dataset, BytedLoaderDatasetSpec):
        raise TypeError(
            "byted_loader requires BytedLoaderDatasetSpec. "
            "The trainer must bypass the native dataset builder before building this dataloader."
        )

    file_type = file_type.lower()
    _validate_v1_knobs(
        mode=mode,
        file_type=file_type,
        worker_subprocess_num=worker_subprocess_num,
        worker_parallel_read_num=worker_parallel_read_num,
        do_sp_split_in_loader=do_sp_split_in_loader,
        gpu_prefetch=gpu_prefetch,
        enable_balance=enable_balance,
        microbatch_balance=microbatch_balance,
    )

    effective_ckpt_dir = ckpt_dir or dataset.ckpt_dir
    effective_save_interval = save_ckpt_interval if save_ckpt_interval > 0 else dataset.save_ckpt_interval
    _guard_no_forbidden_write_root(effective_ckpt_dir, "byted loader ckpt_dir")

    if _TEST_CLIENT_FACTORY is not None:
        config = _make_test_config(
            dataset=dataset,
            file_type=file_type,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            shuffle_algo=shuffle_algo,
            shuffle_shard_nums=shuffle_shard_nums,
            worker_num=worker_num,
            worker_subprocess_num=worker_subprocess_num,
            worker_parallel_read_num=worker_parallel_read_num,
            worker_prefetch_num=worker_prefetch_num,
            client_prefetch_num=client_prefetch_num,
            server_prefetch_num=server_prefetch_num,
            ckpt_dir=effective_ckpt_dir,
            save_ckpt_interval=effective_save_interval,
            enable_ckpt=enable_ckpt,
            enable_batch_db_save=enable_batch_db_save,
            resume_ckpt_path=resume_ckpt_path,
            resume_use_latest_snapshot=resume_use_latest_snapshot,
            allow_transform_failure=allow_transform_failure,
            do_sp_split_in_loader=do_sp_split_in_loader,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            max_seq_len=max_seq_len,
            train_steps=train_steps,
            seed=seed,
            dsp_rotate_interval=dsp_rotate_interval,
            dp_constructor_per_dp=dp_constructor_per_dp,
        )
        client = _TEST_CLIENT_FACTORY(
            config=config,
            transforms_dict=_build_transforms_dict(dataset),
            microbatch_transforms=[SanitizeThenCollate(collate_fn)],
            seq_len_extract=_seq_len_extract,
            total_steps=train_steps,
            start_role_after_iter=start_role_after_iter,
        )
        _log_effective_config(
            config, identity=identity, strict_api_check=False, start_role_after_iter=start_role_after_iter
        )
        return BytedLoaderAdapter(
            client,
            identity=identity,
            ckpt_dir=effective_ckpt_dir,
            total_steps=train_steps,
            enable_batch_db_save=enable_batch_db_save,
            config_summary=vars(config),
        )

    _set_runtime_env_defaults()
    modules = _import_and_check_byted_loader(min_version=min_version, strict=strict_api_check)
    BytedLoaderConfig = modules["BytedLoaderConfig"]
    DataLoaderClient = modules["DataLoaderClient"]
    PrePeekStrategy = modules["PrePeekStrategy"]

    config = BytedLoaderConfig()
    _populate_byted_config(
        config=config,
        dataset=dataset,
        file_type=file_type,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        shuffle_algo=shuffle_algo,
        shuffle_shard_nums=shuffle_shard_nums,
        worker_num=worker_num,
        worker_subprocess_num=worker_subprocess_num,
        worker_parallel_read_num=worker_parallel_read_num,
        worker_prefetch_num=worker_prefetch_num,
        client_prefetch_num=client_prefetch_num,
        server_prefetch_num=server_prefetch_num,
        ckpt_dir=effective_ckpt_dir,
        save_ckpt_interval=effective_save_interval,
        enable_ckpt=enable_ckpt,
        enable_batch_db_save=enable_batch_db_save,
        resume_ckpt_path=resume_ckpt_path,
        resume_use_latest_snapshot=resume_use_latest_snapshot,
        allow_transform_failure=allow_transform_failure,
        do_sp_split_in_loader=do_sp_split_in_loader,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        max_seq_len=max_seq_len,
        train_steps=train_steps,
        seed=seed,
        dsp_rotate_interval=dsp_rotate_interval,
        dp_constructor_per_dp=dp_constructor_per_dp,
    )

    strategy = PrePeekStrategy(
        peek_num=peek_num,
        max_seq_len=max_seq_len,
        seq_len_warmup=seq_len_warmup,
        micro_batch_size=micro_batch_size,
        num_warmup_steps=num_warmup_steps,
        bsz_warmup_init_mbtoken=bsz_warmup_init_mbtoken,
        enable_balance=False,
        microbatch_balance=False,
        biwise_balance=biwise_balance,
    )
    client = DataLoaderClient(
        config=config,
        strategy=strategy,
        transforms_dict=_build_transforms_dict(dataset),
        microbatch_transforms=[SanitizeThenCollate(collate_fn)],
        identity=identity,
        seq_len_extract=_seq_len_extract,
        total_steps=train_steps,
        pin_memory=pin_memory,
        gpu_prefetch=False,
        allow_transform_failure=allow_transform_failure,
        start_role_after_iter=start_role_after_iter,
    )
    _log_effective_config(
        config, identity=identity, strict_api_check=strict_api_check, start_role_after_iter=start_role_after_iter
    )
    return BytedLoaderAdapter(
        client,
        identity=identity,
        ckpt_dir=effective_ckpt_dir,
        total_steps=train_steps,
        enable_batch_db_save=enable_batch_db_save,
        config_summary=_config_to_dict(config),
    )


def _disable_veomni_multisource_meter(args: Any) -> None:
    original = bool(getattr(args.data, "enable_multisource", False))
    args.data._byted_loader_original_enable_multisource = original
    args.data._byted_loader_source_config_path = args.data.train_path if original else ""
    args.data.enable_multisource = False


def _resolve_byted_ckpt_dir(args: Any) -> str:
    configured = getattr(args.data.dataloader, "ckpt_dir", "") or ""
    return configured or args.train.checkpoint.output_dir


def _resolve_save_ckpt_interval(args: Any) -> int:
    configured = int(getattr(args.data.dataloader, "save_ckpt_interval", -1))
    if configured > 0:
        return configured
    save_steps = int(getattr(args.train.checkpoint, "save_steps", 0) or 0)
    return save_steps if save_steps > 0 else 50


def _validate_v1_knobs(
    *,
    mode: str,
    file_type: str,
    worker_subprocess_num: int,
    worker_parallel_read_num: int,
    do_sp_split_in_loader: bool,
    gpu_prefetch: bool,
    enable_balance: bool,
    microbatch_balance: bool,
) -> None:
    if mode != "local":
        raise ValueError("byted_loader v1 supports only mode='local'.")
    if file_type not in {"lance", "jsonl", "parquet"}:
        raise ValueError(f"Unsupported byted_loader file_type={file_type!r}; expected lance/jsonl/parquet.")
    if file_type == "parquet" and (worker_subprocess_num != 1 or worker_parallel_read_num != 1):
        raise ValueError(
            "byted_loader parquet is restricted compatibility mode: set worker_subprocess_num=1 and "
            "worker_parallel_read_num=1, or convert/use Lance."
        )
    if do_sp_split_in_loader:
        raise ValueError("byted_loader v1 requires do_sp_split_in_loader=false; VeOmni MainCollator owns SP.")
    if gpu_prefetch:
        raise ValueError("byted_loader v1 requires gpu_prefetch=false; CPU tensor path is the supported mode.")
    if enable_balance or microbatch_balance:
        raise ValueError("byted_loader v1 keeps sample/microbatch balance disabled for correctness.")

    ps = get_parallel_state()
    if getattr(ps, "pp_size", 1) != 1 or getattr(ps, "tp_size", 1) != 1:
        raise ValueError(
            "byted_loader v1 supports pp_size=1 and tp_size=1 only. "
            f"Got pp_size={getattr(ps, 'pp_size', 'n/a')} tp_size={getattr(ps, 'tp_size', 'n/a')}."
        )


def _import_and_check_byted_loader(*, min_version: str, strict: bool) -> Dict[str, Any]:
    try:
        version = importlib.metadata.version("bytedance.dataloader")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "data.dataloader.type=byted_loader requires package `bytedance.dataloader`. "
            "Install it in the training image, for example: pip3 install bytedance.dataloader --no-build-isolation"
        ) from exc

    if _version_tuple(version) < _version_tuple(min_version):
        raise ImportError(f"bytedance.dataloader>={min_version} required, found {version}.")

    config_mod = importlib.import_module("bytedance.dataloader.config")
    client_mod = importlib.import_module("bytedance.dataloader.local_mode.client")
    strategy_mod = importlib.import_module("bytedance.dataloader.strategy.pre_peek")
    modules = {
        "version": version,
        "BytedLoaderConfig": config_mod.BytedLoaderConfig,
        "DataLoaderClient": client_mod.DataLoaderClient,
        "PrePeekStrategy": strategy_mod.PrePeekStrategy,
    }
    if strict:
        _check_api_contract(modules)
    return modules


def _set_runtime_env_defaults() -> None:
    os.environ.setdefault("BYTED_LOADER_USE_FUSE_READER", "0")


def _check_api_contract(modules: Mapping[str, Any]) -> None:
    config_fields = set(getattr(modules["BytedLoaderConfig"], "__annotations__", {}))
    required_config = {
        "train_path",
        "file_type",
        "ckpt_dir",
        "save_ckpt_interval",
        "enable_batch_db_save",
        "worker_subprocess_num",
        "worker_parallel_read_num",
        "do_sp_split_in_loader",
        "dp_rank",
        "dp_size",
        "dsp_rank",
        "dsp_size",
    }
    missing = sorted(required_config - config_fields)
    if missing:
        raise RuntimeError(f"BytedLoaderConfig API missing required fields: {', '.join(missing)}")

    client_params = set(inspect.signature(modules["DataLoaderClient"]).parameters)
    required_client = {"config", "strategy", "transforms_dict", "microbatch_transforms", "identity", "seq_len_extract"}
    missing = sorted(required_client - client_params)
    if missing:
        raise RuntimeError(f"DataLoaderClient API missing required parameters: {', '.join(missing)}")

    strategy_params = set(inspect.signature(modules["PrePeekStrategy"]).parameters)
    required_strategy = {"peek_num", "max_seq_len", "enable_balance", "microbatch_balance"}
    missing = sorted(required_strategy - strategy_params)
    if missing:
        raise RuntimeError(f"PrePeekStrategy API missing required parameters: {', '.join(missing)}")


def _populate_byted_config(config: Any, **kwargs: Any) -> None:
    for key, value in _make_config_values(**kwargs).items():
        setattr(config, key, value)


def _make_test_config(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**_make_config_values(**kwargs))


def _make_config_values(
    *,
    dataset: BytedLoaderDatasetSpec,
    file_type: str,
    shuffle: Optional[bool],
    shuffle_seed: Optional[int],
    shuffle_algo: str,
    shuffle_shard_nums: int,
    worker_num: int,
    worker_subprocess_num: int,
    worker_parallel_read_num: int,
    worker_prefetch_num: int,
    client_prefetch_num: int,
    server_prefetch_num: int,
    ckpt_dir: str,
    save_ckpt_interval: int,
    enable_ckpt: bool,
    enable_batch_db_save: bool,
    resume_ckpt_path: str,
    resume_use_latest_snapshot: bool,
    allow_transform_failure: bool,
    do_sp_split_in_loader: bool,
    micro_batch_size: int,
    global_batch_size: int,
    max_seq_len: int,
    train_steps: int,
    seed: int,
    dsp_rotate_interval: int,
    dp_constructor_per_dp: int,
) -> Dict[str, Any]:
    ps = get_parallel_state()
    rank = _rank()
    local_rank = _local_rank()
    world_size = _world_size(ps)
    local_world_size = _local_world_size()
    worker_rank_interval = 4  # bytedance.dataloader 0.1.42 local mode role-builder default.
    per_node_worker_num = max(local_world_size // worker_rank_interval, 1)
    node_num = max(world_size // max(local_world_size, 1), 1)
    auto_worker_num = node_num * per_node_worker_num
    effective_shuffle_seed = shuffle_seed
    if effective_shuffle_seed is None:
        effective_shuffle_seed = dataset.shuffle_seed if dataset.shuffle_seed is not None else seed

    return {
        "total_steps": train_steps,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "max_seq_len": max_seq_len,
        "shuffle": dataset.shuffle if shuffle is None else bool(shuffle),
        "shuffle_seed": int(effective_shuffle_seed),
        "shuffle_algo": shuffle_algo,
        "shuffle_shard_nums": int(shuffle_shard_nums or dataset.shuffle_shard_nums),
        "file_type": file_type,
        "ckpt_dir": ckpt_dir,
        "save_ckpt_interval": save_ckpt_interval,
        "enable_ckpt": enable_ckpt,
        "enable_batch_db_save": enable_batch_db_save,
        "drop_resume_buffer": False,
        "train_path": dataset.train_path,
        "worker_subprocess_num": worker_subprocess_num,
        "worker_parallel_read_num": worker_parallel_read_num,
        "worker_prefetch_num": worker_prefetch_num,
        "client_prefetch_num": client_prefetch_num,
        "server_prefetch_num": server_prefetch_num,
        "worker_num": auto_worker_num if worker_num < 1 else worker_num,
        "resume_ckpt_path": resume_ckpt_path,
        "resume_use_latest_snapshot": resume_use_latest_snapshot,
        "allow_transform_failure": allow_transform_failure,
        "do_sp_split_in_loader": do_sp_split_in_loader,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "client_num": world_size,
        "dp_rank": _parallel_attr(ps, "dp_rank", 0),
        "dp_size": _parallel_attr(ps, "dp_size", 1),
        "tp_rank": _parallel_attr(ps, "tp_rank", 0),
        "tp_size": _parallel_attr(ps, "tp_size", 1),
        "pp_rank": _parallel_attr(ps, "pp_rank", 0),
        "pp_size": _parallel_attr(ps, "pp_size", 1),
        "dsp_rank": _parallel_attr(ps, "sp_rank", 0),
        "dsp_size": _parallel_attr(ps, "sp_size", 1),
        "dsp_rotate_interval": dsp_rotate_interval,
        "dp_constructor_per_dp": dp_constructor_per_dp,
        "trainer_strategy": "fsdp",
    }


def _build_transforms_dict(dataset: BytedLoaderDatasetSpec) -> Mapping[str, List[Callable[[Any], Any]]]:
    return _DefaultTransformMapping("text_transform", [VeOmniSampleTransform(dataset.transform)])


def _sanitize_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in sample.items()
        if not str(key).startswith("byted_") and key not in _LOADER_METADATA_KEYS
    }


def _callable_accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _seq_len_extract(sample: Mapping[str, Any]) -> int:
    attention_mask = sample.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.sum().item())
    if attention_mask is not None:
        return int(sum(attention_mask))

    input_ids = sample.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    if input_ids is not None:
        return int(len(input_ids))
    raise KeyError("Cannot extract seq len: sample has neither attention_mask nor input_ids.")


def _is_byted_local_mode_client(client: Any) -> bool:
    return type(client).__module__.startswith("bytedance.dataloader.local_mode.")


def _force_close_byted_local_mode_client(client: Any, identity: str) -> None:
    stop_event = getattr(client, "stop_event", None)
    if hasattr(stop_event, "set"):
        stop_event.set()

    rank_role_builder = getattr(client, "rank_role_builder", None)
    if hasattr(rank_role_builder, "exit"):
        try:
            rank_role_builder.exit()
        except Exception as exc:
            logger.warning_rank0(f"[byted_loader] failed to terminate local roles: {exc}")

    prefetch_queue = getattr(client, "prefetch_queue", None)
    if prefetch_queue is not None:
        try:
            while not prefetch_queue.empty():
                prefetch_queue.get_nowait()
        except Exception:
            pass

    async_task_pool = getattr(client, "async_task_pool", None)
    if async_task_pool is not None:
        try:
            async_task_pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            async_task_pool.shutdown(wait=False)
        except Exception as exc:
            logger.warning_rank0(f"[byted_loader] failed to shutdown client executor: {exc}")

    try:
        clear_callback_state = importlib.import_module(
            "bytedance.dataloader.local_mode.byterpc.client"
        ).clear_callback_state
        clear_callback_state(identity)
    except Exception:
        pass

    try:
        importlib.import_module("veturborpc.rpc").shutdown()
    except Exception:
        pass


def _rank() -> int:
    try:
        ps = get_parallel_state()
        if getattr(ps, "global_rank", -1) >= 0:
            return int(ps.global_rank)
    except Exception:
        pass
    return int(os.getenv("RANK", "0"))


def _local_rank() -> int:
    try:
        ps = get_parallel_state()
        if getattr(ps, "local_rank", -1) >= 0:
            return int(ps.local_rank)
    except Exception:
        pass
    return int(os.getenv("LOCAL_RANK", "0"))


def _parallel_attr(ps: Any, name: str, default: int) -> int:
    try:
        value = int(getattr(ps, name))
    except Exception:
        return default
    return default if value < 0 else value


def _world_size(ps: Any) -> int:
    value = int(getattr(ps, "world_size", 1))
    if value > 0:
        return value
    return int(os.getenv("WORLD_SIZE", "1"))


def _local_world_size() -> int:
    return max(int(os.getenv("LOCAL_WORLD_SIZE", os.getenv("ARNOLD_WORKER_GPU", "1"))), 1)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: List[int] = []
    for part in version.replace("-", ".").split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            digits = "".join(ch for ch in part if ch.isdigit())
            parts.append(int(digits or 0))
    return tuple(parts)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {name: getattr(config, name) for name in dir(config) if not name.startswith("_")}


def _log_effective_config(config: Any, *, identity: str, strict_api_check: bool, start_role_after_iter: bool) -> None:
    summary = {
        key: getattr(config, key, None)
        for key in (
            "train_path",
            "file_type",
            "shuffle",
            "shuffle_algo",
            "shuffle_seed",
            "global_batch_size",
            "micro_batch_size",
            "max_seq_len",
            "rank",
            "world_size",
            "dp_rank",
            "dp_size",
            "dsp_rank",
            "dsp_size",
            "tp_size",
            "pp_size",
            "worker_num",
            "worker_subprocess_num",
            "worker_parallel_read_num",
            "ckpt_dir",
            "save_ckpt_interval",
            "enable_ckpt",
            "enable_batch_db_save",
            "do_sp_split_in_loader",
        )
    }
    summary["start_role_after_iter"] = start_role_after_iter
    summary["byted_loader_use_fuse_reader_env"] = os.environ.get("BYTED_LOADER_USE_FUSE_READER")
    logger.info_rank0(
        "[byted_loader] effective_config "
        f"identity={identity} strict_api_check={strict_api_check} "
        "disable_veomni_multisource_meter=true "
        f"config={summary}"
    )
    if getattr(config, "file_type", "") == "parquet":
        logger.warning_rank0(
            "[byted_loader] parquet is restricted compatibility mode. Use Lance for real dryrun/performance validation."
        )


def _audit_enabled() -> bool:
    return os.environ.get("VEOMNI_BYTED_LOADER_AUDIT", "").strip().lower() in {"1", "true", "yes", "on"}


def _audit_limit() -> int:
    try:
        return int(os.environ.get("VEOMNI_BYTED_LOADER_AUDIT_STEPS", "0"))
    except ValueError:
        return 0


def _audit_file():
    global _AUDIT_FILE
    if _AUDIT_FILE is not None:
        return _AUDIT_FILE
    audit_dir = Path(os.environ.get("VEOMNI_BYTED_LOADER_AUDIT_DIR", "/tmp/veomni_byted_loader_audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    _AUDIT_FILE = open(audit_dir / f"byted_adapter_rank{_rank()}_pid{os.getpid()}.jsonl", "a", buffering=1)
    return _AUDIT_FILE


def _log_adapter_yield(
    *,
    identity: str,
    yield_index: int,
    duration_ms: float,
    batch: Any,
    config_summary: Mapping[str, Any],
    audit_limit: int,
) -> None:
    if audit_limit > 0 and yield_index > audit_limit:
        return
    payload = {
        "event": "byted_adapter_yield",
        "time": time.time(),
        "rank": _rank(),
        "identity": identity,
        "yield_index": yield_index,
        "duration_ms": round(float(duration_ms), 6),
        "microbatch_count": len(batch) if isinstance(batch, list) else 1,
        "fingerprint": _batch_fingerprint(batch),
        "config": {
            key: config_summary.get(key)
            for key in (
                "file_type",
                "shuffle",
                "shuffle_algo",
                "global_batch_size",
                "micro_batch_size",
                "max_seq_len",
                "dp_size",
                "dsp_size",
                "worker_num",
                "worker_subprocess_num",
                "worker_parallel_read_num",
                "enable_batch_db_save",
                "do_sp_split_in_loader",
            )
        },
    }
    try:
        _audit_file().write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except Exception as exc:
        logger.warning_rank0(f"[byted_loader] failed to write adapter audit: {exc}")


def _batch_fingerprint(batch: Any) -> str:
    digest = hashlib.sha256()
    for microbatch in batch if isinstance(batch, list) else [batch]:
        if not isinstance(microbatch, Mapping):
            digest.update(type(microbatch).__name__.encode())
            continue
        for key in ("input_ids", "labels", "attention_mask"):
            value = microbatch.get(key)
            digest.update(str(key).encode())
            if torch.is_tensor(value):
                detached = value.detach()
                digest.update(str(tuple(int(dim) for dim in detached.shape)).encode())
                try:
                    digest.update(str(int(detached.sum().item())).encode())
                    if detached.numel() > 0:
                        flat = detached.reshape(-1)
                        digest.update(str(int(flat[0].item())).encode())
                        digest.update(str(int(flat[-1].item())).encode())
                except Exception:
                    digest.update(str(detached.dtype).encode())
            else:
                digest.update(str(value).encode())
    return digest.hexdigest()[:16]


def _guard_no_forbidden_write_root(path: str, label: str) -> None:
    for root in _forbidden_write_roots():
        if root in str(path):
            raise ValueError(f"{label} points to forbidden write root: {path}")


def _forbidden_write_roots() -> tuple[str, ...]:
    raw = os.environ.get("VEOMNI_FORBIDDEN_WRITE_ROOTS", "")
    return tuple(root.strip() for root in raw.split(",") if root.strip())
