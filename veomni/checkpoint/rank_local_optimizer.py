# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Exact-rank optimizer DCP keys for fixed-geometry recovery experiments.

Unlike globally named scalar steps, rank-local keys cannot be deduplicated
across different expert owners. This intentionally trades checkpoint storage
for exact state preservation and requires an independently verified rank and
mesh identity before use. It does not support elastic resharding.
"""

import hashlib
import heapq
import io
import json
import struct
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import BytesStorageMetadata
from torch.distributed.tensor import DTensor


def rank_prefix(rank: int) -> str:
    if type(rank) is not int or rank < 0:
        raise ValueError("optimizer checkpoint rank must be a non-negative integer")
    return f"__veomni_rank_{rank}__"


def rank_scalar_key(rank: int) -> str:
    return rank_prefix(rank) + "__scalars"


def _is_small_optimizer_value(key: str) -> bool:
    return key.startswith("param_groups.") or (key.startswith("state.") and key.endswith(".step"))


def _decode_scalar_payload(payload) -> dict:
    if not isinstance(payload, bytes):
        raise RuntimeError("rank-local optimizer scalar payload is not bytes")
    values = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not _is_small_optimizer_value(key) for key in values
    ):
        raise RuntimeError("rank-local optimizer scalar payload is malformed")
    return values


def local_optimizer_state(state: dict, rank: int) -> dict:
    """Expose physical shards under rank-unique keys; pack steps and group options."""
    prefix = rank_prefix(rank)
    local, small, pending = {}, {}, {}
    for key, value in state.items():
        if not isinstance(key, str) or key.startswith("__veomni_rank_"):
            raise ValueError("invalid optimizer checkpoint key")
        if _is_small_optimizer_value(key):
            if isinstance(value, DTensor):
                raise RuntimeError("rank-local optimizer scalar cannot be a DTensor")
            if torch.is_tensor(value) and value.device.type != "cpu":
                if value.ndim != 0:
                    raise RuntimeError("rank-local accelerator optimizer metadata must be scalar")
                pending.setdefault((value.device, value.dtype), []).append((key, value.detach()))
            else:
                small[key] = value.detach() if torch.is_tensor(value) else value
            continue
        if isinstance(value, DTensor):
            value = value.to_local()
        local[prefix + key] = value
    for entries in pending.values():
        transferred = torch.stack([value for _, value in entries]).cpu()
        for index, (key, _) in enumerate(entries):
            small[key] = transferred[index]
    buffer = io.BytesIO()
    torch.save(small, buffer)
    local[rank_scalar_key(rank)] = buffer.getvalue()
    return local


def rank_inventory(state: dict, rank: int) -> list[str]:
    flattened, _ = flatten_state_dict({"optimizer": local_optimizer_state(state, rank)})
    return sorted(flattened)


def rank_inventory_key(rank: int) -> str:
    return rank_prefix(rank) + "__inventory"


def load_rank_scalars(reader_factory, checkpoint_path, rank):
    key = rank_scalar_key(rank)
    reader = reader_factory(checkpoint_path)
    metadata = reader.read_metadata()
    if not isinstance(metadata.state_dict_metadata.get(f"optimizer.{key}"), BytesStorageMetadata):
        raise RuntimeError("rank-local optimizer scalar payload is missing")
    scratch = {"optimizer": {key: b""}}
    dcp.load(scratch, storage_reader=reader_factory(checkpoint_path), no_dist=True)
    return _decode_scalar_payload(scratch["optimizer"][key])


def restore_optimizer_state(local: dict, template: dict, rank: int) -> dict:
    """Restore the DTensor wrappers from the already-allocated optimizer state."""
    prefix = rank_prefix(rank)
    expected = {prefix + key for key in template if not _is_small_optimizer_value(key)} | {rank_scalar_key(rank)}
    if set(local) != expected:
        raise RuntimeError("rank-local optimizer checkpoint inventory differs")
    small = _decode_scalar_payload(local[rank_scalar_key(rank)])
    if set(small) != {key for key in template if _is_small_optimizer_value(key)}:
        raise RuntimeError("rank-local optimizer scalar inventory differs")
    result = {}
    for key, expected in template.items():
        value = small[key] if _is_small_optimizer_value(key) else local[prefix + key]
        if _is_small_optimizer_value(key) and torch.is_tensor(expected):
            if (
                not torch.is_tensor(value)
                or expected.dtype != value.dtype
                or tuple(expected.shape) != tuple(value.shape)
            ):
                raise RuntimeError("rank-local optimizer scalar shape or dtype differs")
            expected.copy_(value.to(device=expected.device))
            result[key] = expected
            continue
        if isinstance(expected, DTensor):
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("rank-local optimizer tensor was replaced with a non-tensor")
            original = expected.to_local()
            if value.dtype != original.dtype or tuple(value.shape) != tuple(original.shape):
                raise RuntimeError("rank-local optimizer tensor shape or dtype differs")
            if value.data_ptr() != original.data_ptr():
                original.copy_(value)
            result[key] = expected
        else:
            result[key] = value
    return result


def current_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def wrap_local_like_parameter(local, param):
    """Preserve a parameter's distributed geometry for a local state shard."""
    if not isinstance(param, DTensor):
        return local
    if local.device.type == "cpu" and param.device_mesh.device_type != "cpu":
        metadata = DTensor.from_local(
            local.to("meta"),
            param.device_mesh,
            param.placements,
            run_check=False,
            shape=param.shape,
            stride=param.stride(),
        )
        return DTensor(local, metadata._spec, requires_grad=False)
    return DTensor.from_local(
        local, param.device_mesh, param.placements, run_check=False, shape=param.shape, stride=param.stride()
    )


def zeros_like_parameter_on_cpu(param, *, dtype):
    """Allocate one local optimizer shard on CPU with the parameter's DTensor spec."""
    local = param.to_local() if isinstance(param, DTensor) else param
    return wrap_local_like_parameter(torch.zeros_like(local, dtype=dtype, device="cpu"), param)


def _reject_overlapping_geometries(name, geometries):
    """Reject distinct overlapping boxes before DCP deduplicates by offset."""
    offsets_to_size = {}
    nonempty = []
    for offsets, size in geometries:
        old_size = offsets_to_size.setdefault(offsets, size)
        if old_size != size:
            raise RuntimeError(f"Checkpoint tensor {name} has overlapping nonidentical shards")
        if all(size):
            nonempty.append((offsets, size))
    if len(nonempty) < 2 or not nonempty[0][0]:
        return
    # Choose a dimension with the most distinct intervals to keep the sweep
    # active set small for both row- and column-sharded tensors.
    axis = max(
        range(len(nonempty[0][0])),
        key=lambda dim: len({(offsets[dim], size[dim]) for offsets, size in nonempty}),
    )
    ordered = sorted(nonempty, key=lambda item: item[0][axis])
    active, expiration = {}, []
    for index, (offsets, size) in enumerate(ordered):
        start = offsets[axis]
        while expiration and expiration[0][0] <= start:
            _, previous = heapq.heappop(expiration)
            del active[previous]
        for other_offsets, other_size in active.values():
            if all(
                start < other_start + other_length and other_start < start + length
                for start, length, other_start, other_length in zip(
                    offsets, size, other_offsets, other_size, strict=True
                )
            ):
                raise RuntimeError(f"Checkpoint tensor {name} has overlapping nonidentical shards")
        active[index] = (offsets, size)
        heapq.heappush(expiration, (start + size[axis], index))


def verify_replicated_model_shards(state: dict, *, group=None, batch_size=128, chunk_bytes=4 << 20):
    """Reject divergent model replicas before DCP deduplicates their shards.

    Hashing is chunked to bound temporary CPU/NPU memory, including for
    non-contiguous model shards.
    """
    from ..utils.dist_utils import raise_if_any_rank_failed

    def scalar_bytes(value):
        if type(value) is float:
            return b"float:" + struct.pack("!d", value)
        if type(value) in (str, int, bool, bytes, type(None), torch.dtype, torch.device):
            return type(value).__name__.encode() + b":" + repr(value).encode()
        if type(value) in (tuple, list):
            parts = [scalar_bytes(item) for item in value]
            return (
                type(value).__name__.encode() + b":" + b"".join(len(part).to_bytes(8, "big") + part for part in parts)
            )
        raise RuntimeError(f"Cannot prove non-tensor checkpoint state of type {type(value).__name__}")

    error, records = None, []
    try:
        flattened, _ = flatten_state_dict({"model": state})
        for name, value in sorted(flattened.items()):
            if not torch.is_tensor(value):
                payload = scalar_bytes(value)
                records.append((name, (), (), type(value).__name__, hashlib.sha256(payload).hexdigest(), None))
                continue
            if isinstance(value, DTensor):
                chunks = value.__create_chunk_list__()
                if len(chunks) != 1:
                    raise RuntimeError(f"Cannot prove model shard geometry for {name}")
                offsets, size = tuple(chunks[0].offsets), tuple(chunks[0].sizes)
                local = value.to_local()
            else:
                local = value
                offsets, size = (0,) * value.ndim, tuple(value.shape)
            if tuple(local.shape) != size:
                raise RuntimeError(f"Cannot prove model shard shape for {name}")
            from .tensor_bytes import tensor_byte_chunks

            digest = hashlib.sha256()
            for chunk in tensor_byte_chunks(local, chunk_bytes=chunk_bytes):
                digest.update(chunk)
            records.append((name, offsets, size, str(local.dtype), digest.hexdigest(), tuple(value.shape)))
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "hashing replicated model shards", group=group)
    if not dist.is_initialized():
        return
    names = hashlib.sha256("\0".join(item[0] for item in records).encode()).hexdigest()
    identities = [None] * dist.get_world_size(group)
    dist.all_gather_object(identities, (len(records), names), group=group)
    if any(identity != identities[0] for identity in identities):
        raise RuntimeError("Model ranks have different checkpoint tensor inventories")
    for start in range(0, len(records), batch_size):
        gathered = [None] * len(identities)
        dist.all_gather_object(gathered, records[start : start + batch_size], group=group)
        seen = {}
        geometries = {}
        global_signatures = {}
        for items in gathered:
            for name, offsets, size, dtype, digest, global_shape in items:
                signature = (dtype, global_shape)
                previous = global_signatures.setdefault(name, signature)
                if previous != signature:
                    raise RuntimeError(f"Checkpoint tensor {name} has different global dtype or shape across ranks")
                identity = (name, offsets, size)
                if identity not in seen:
                    geometries.setdefault(name, []).append((offsets, size))
                old = seen.setdefault(identity, (dtype, digest))
                if old != (dtype, digest):
                    raise RuntimeError(f"Model replicas differ for checkpoint tensor {name} at {offsets}")
        for name, shards in geometries.items():
            _reject_overlapping_geometries(name, shards)


@contextmanager
def recover_frozen_optimizer_ownership(model, checkpoint_path, *, module="", group=None):
    """Include source-recorded frozen parameters while building an optimizer.

    The original flags are restored before the first forward. A later layout
    proof still checks the reconstructed optimizer groups against the source.
    """
    from torch.distributed.checkpoint.state_dict import _get_fqns

    from ..utils.dist_utils import raise_if_any_rank_failed
    from .dcp_checkpointer import DistributedCheckpointer
    from .layout import optimizer_dir
    from .multi_optimizer_checkpoint import MULTI_LAYOUT_MARKER

    changed, error = [], None
    try:
        path = optimizer_dir(checkpoint_path, module)
        reader = DistributedCheckpointer._create_storage_reader(path)
        metadata = reader.read_metadata()
        if not isinstance(metadata.state_dict_metadata.get(f"optimizer.{MULTI_LAYOUT_MARKER}"), BytesStorageMetadata):
            raise RuntimeError("Cannot recover frozen ownership from an unproven optimizer checkpoint")
        rank = current_rank()
        prefix = f"optimizer.{rank_prefix(rank)}"
        inventory_path = "optimizer." + rank_inventory_key(rank)
        if isinstance(metadata.state_dict_metadata.get(inventory_path), BytesStorageMetadata):
            scratch = {"optimizer": {rank_inventory_key(rank): ""}}
            dcp.load(scratch, storage_reader=DistributedCheckpointer._create_storage_reader(path), no_dist=True)
            inventory = json.loads(scratch["optimizer"][rank_inventory_key(rank)])
            actual = sorted(k for k in metadata.state_dict_metadata if k.startswith(prefix) and k != inventory_path)
            if inventory != actual:
                raise RuntimeError("Optimizer checkpoint ownership inventory differs from DCP metadata")
            owned_keys = set(load_rank_scalars(DistributedCheckpointer._create_storage_reader, path, rank))
        else:
            owned_keys = {
                key.removeprefix("optimizer.")
                for key in metadata.state_dict_metadata
                if key.startswith("optimizer.param_groups.")
            }
        for name, param in model.named_parameters(remove_duplicate=False):
            if not param.requires_grad and any(
                f"param_groups.{fqn}.lr" in owned_keys for fqn in _get_fqns(model, name)
            ):
                if all(param is not existing for existing in changed):
                    param.requires_grad_(True)
                    changed.append(param)
    except BaseException as exc:
        error = exc
    try:
        raise_if_any_rank_failed(error, "recovering frozen optimizer ownership", group=group)
        yield
    finally:
        for param in changed:
            param.requires_grad_(False)
