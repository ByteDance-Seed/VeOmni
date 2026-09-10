# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared offline DCP reading and shard writing for checkpoint exporters."""

import gc
import json
import os
import struct
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from veomni.utils import helper
from veomni.utils.device import empty_cache


logger = helper.create_logger(__name__)


class CachedFileSystemReader(FileSystemReader):
    """Read DCP metadata once across all shard loads."""

    def __init__(self, path):
        super().__init__(path)
        self._cached_metadata = None

    def read_metadata(self):
        if self._cached_metadata is None:
            self._cached_metadata = super().read_metadata()
        return self._cached_metadata


def load_source_tensors(reader, keys: Mapping[str, str]) -> dict[str, torch.Tensor]:
    metadata = reader.read_metadata()
    state_dict = OrderedDict()
    for dcp_key in keys.values():
        meta = metadata.state_dict_metadata[dcp_key]
        state_dict[dcp_key] = torch.empty(meta.size, dtype=meta.properties.dtype)
    dcp_load(state_dict, storage_reader=reader, no_dist=True)
    return state_dict


def hf_tensors(keys, source, save_dtype="bfloat16"):
    """Normalize keys and cast floating tensors, preserving integer/bool buffers."""
    dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
    result = OrderedDict()
    for hf_key, dcp_key in keys.items():
        tensor = source.pop(dcp_key)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        result[hf_key] = tensor.detach().cpu().contiguous().clone()
    return result


def write_shard(output_dir, filename, tensors, *, safe_serialization=True, metadata=None):
    path = os.path.join(output_dir, filename)
    temporary = path + ".tmp"
    if safe_serialization:
        save_file(tensors, temporary, metadata=metadata)
    else:
        torch.save(tensors, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def export_shards(
    groups: Sequence[tuple[str, Mapping[str, str]]],
    reader: CachedFileSystemReader,
    output_dir: str,
    transform: Callable,
    *,
    safe_serialization: bool = True,
    shard_metadata: Mapping | None = None,
    skip_shard: Callable | None = None,
) -> tuple[dict[str, str], int]:
    """Read, transform and atomically write one shard at a time.

    The format-specific transform consumes DCP-keyed tensors and returns output-keyed
    tensors. Return the actual output key map and tensor byte count for index writing.
    """
    os.makedirs(output_dir, exist_ok=True)
    weight_map = OrderedDict()
    total_size = 0
    for index, (filename, keys) in enumerate(groups, start=1):
        if skip_shard is not None and skip_shard(filename):
            with open(os.path.join(output_dir, filename), "rb") as fh:
                length = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(length))
            header.pop("__metadata__", None)
            for key, entry in header.items():
                weight_map[key] = filename
                total_size += entry["data_offsets"][1] - entry["data_offsets"][0]
            logger.info(f"[{index}/{len(groups)}] {filename}: already complete, skipping")
            continue
        started = time.monotonic()
        source = load_source_tensors(reader, keys)
        tensors = transform(keys, source)
        del source
        write_shard(
            output_dir,
            filename,
            tensors,
            safe_serialization=safe_serialization,
            metadata=shard_metadata.get(filename) if shard_metadata is not None else {"format": "pt"},
        )
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        weight_map.update(dict.fromkeys(tensors, filename))
        del tensors
        gc.collect()
        empty_cache()
        logger.info(f"[{index}/{len(groups)}] {filename}: done in {time.monotonic() - started:.1f}s")
    return weight_map, total_size


def write_index(output_dir, weight_map, total_size, filename="model.safetensors.index.json"):
    path = os.path.join(output_dir, filename)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(path + ".tmp", path)
