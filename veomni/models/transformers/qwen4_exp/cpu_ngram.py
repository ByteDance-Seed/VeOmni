"""Opt-in frozen CPU PLE storage for the Qwen4-Exp training.

The caller owns distributed routing and checkpoint identity. Only already-local
row/column shards are accepted: this module never gathers a vocabulary table.
Installed after HF loading/FSDP wrapping, before optimizer creation. Checkpoint
save/resume uses synchronous full-model sharded DCP; NPU qualification is profile-specific.
"""

import logging
from contextlib import nullcontext
from types import MethodType

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


logger = logging.getLogger(__name__)


class FrozenCpuNgramLookup:
    """Own immutable CPU copies of local table shards, with no parameter grads."""

    def __init__(self, local_weights, *, copy_weights=True):
        weights = tuple(local_weights)
        if not weights:
            raise ValueError("At least one local table is required")
        for weight in weights:
            if hasattr(weight, "to_local"):
                raise ValueError("Pass explicit local tensors, not DTensors")
            if weight.ndim != 2 or weight.is_meta or not weight.is_floating_point():
                raise ValueError("Expected materialized floating-point matrix shards")
            if weight.requires_grad:
                raise ValueError("Freeze ngram weights explicitly before CPU lookup")
            if weight.shape[1] != weights[0].shape[1] or weight.dtype != weights[0].dtype:
                raise ValueError("Local shards must have identical column count and dtype")
        self._weights = tuple(w.detach().to(device="cpu", copy=copy_weights).contiguous() for w in weights)

    @torch.no_grad()
    def __call__(self, shard_ids, row_ids, *, output_dtype=None):
        if shard_ids.device != row_ids.device:
            raise ValueError("Request tensors must share a device")
        for ids in (shard_ids, row_ids):
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Requests must be one-dimensional integer tensors")
        if shard_ids.shape != row_ids.shape:
            raise ValueError("Request lengths differ")
        destination = row_ids.device
        observer = getattr(self, "timing", None)
        scope = observer.span if observer is not None else lambda _: nullcontext()
        with scope("ple_ids_d2h"):
            shard_ids = shard_ids.to(device="cpu", dtype=torch.int64)
            row_ids = row_ids.to(device="cpu", dtype=torch.int64)
        if ((shard_ids < 0) | (shard_ids >= len(self._weights))).any():
            raise ValueError("Invalid table shard ID")
        dtype = output_dtype or self._weights[0].dtype
        if not dtype.is_floating_point:
            raise ValueError("Output must be floating point")
        result = torch.empty((row_ids.numel(), self._weights[0].shape[1]), dtype=dtype)
        with scope("ple_cpu_lookup"):
            for index, weight in enumerate(self._weights):
                positions = torch.where(shard_ids == index)[0]
                rows = row_ids[positions]
                if ((rows < 0) | (rows >= weight.shape[0])).any():
                    raise ValueError("Invalid local row ID")
                result[positions] = torch.nn.functional.embedding(rows, weight).to(dtype)
        if observer is not None and observer.active:
            observer.bytes["ple_ids_d2h"] += 2 * row_ids.numel() * 8
            observer.bytes["ple_result_h2d"] += result.numel() * result.element_size()
        with scope("ple_result_h2d"):
            return result.to(destination)


def _ngram_modules(model):
    return [(name, module) for name, module in model.named_modules() if name.endswith(".ple.ple_embedding")]


def validate_cpu_ngram_config(args):
    """Reject unsupported combinations before loading the full checkpoint."""
    ckpt = args.train.checkpoint
    if ckpt.manager != "dcp" or ckpt.save_hf_weights or ckpt.save_async:
        raise ValueError("CPU PLE supports synchronous DCP only; HF and async saves are unsupported")
    if args.model.accelerator.fsdp_config.fsdp_mode != "fsdp2":
        raise ValueError("CPU PLE requires FSDP2 persistent 2D sharding")
    if args.model.accelerator.torch_compile.enable:
        raise ValueError("CPU PLE does not support torch.compile")


def freeze_ngram_tables(model):
    """Freeze only vocabulary tables, not PLE projections/gates."""
    if model.config.model_type != "qwen4_exp":
        raise ValueError("CPU PLE requires qwen4_exp")
    modules = _ngram_modules(model)
    if not modules:
        raise ValueError("No Qwen4-Exp PLE modules found")
    for _, module in modules:
        module.ngram_embedding.requires_grad_(False)


def _cpu_local_lookup(self, shard_ids, row_ids, output_dtype=None):
    return self._cpu_ngram_lookup(shard_ids, row_ids, output_dtype=output_dtype)


def install_cpu_ngram(model):
    """Move only persistent local shards; leave NPU request routing unchanged.

    CPU meshes carry tensor identity, not a second communication backend. No
    collective is issued on them. The existing PLE NPU group routes requests
    and responses using the original rank table.
    """
    modules = _ngram_modules(model)
    if not modules:
        raise ValueError("No PLE module to install")
    persistent_ids = getattr(model, "_persistent_extra_parallel_param_ids", set())
    for name, module in modules:
        if hasattr(module, "_cpu_ngram_lookup"):
            raise ValueError("CPU PLE already installed")
        for embedding in module.ngram_embedding.values():
            p = embedding.weight
            if not isinstance(p, DTensor) or p.placements != (Shard(1), Shard(0)):
                raise ValueError(f"{name}: expected persistent column/row DTensor")
            if id(p) not in persistent_ids or p.requires_grad or p.is_meta:
                raise ValueError(f"{name}: expected frozen, loaded, FSDP-ignored table")
            if p.device_mesh.ndim != 2:
                raise ValueError("CPU PLE requires a 2D mesh")
    cpu_meshes = {}
    moved_bytes = 0
    for _name, module in modules:
        locals_ = []
        for embedding in module.ngram_embedding.values():
            old = embedding.weight
            mesh = old.device_mesh
            if id(mesh) not in cpu_meshes:
                cpu_meshes[id(mesh)] = DeviceMesh(
                    "cpu", mesh.mesh.clone(), mesh_dim_names=mesh.mesh_dim_names, _init_backend=False
                )
            local = old.to_local().detach().to(device="cpu", copy=True).contiguous()
            cpu_weight = torch.nn.Parameter(
                DTensor.from_local(
                    local,
                    cpu_meshes[id(mesh)],
                    old.placements,
                    run_check=False,
                    shape=old.shape,
                    stride=old.stride(),
                ),
                requires_grad=False,
            )
            persistent_ids.remove(id(old))
            persistent_ids.add(id(cpu_weight))
            embedding.weight = cpu_weight
            moved_bytes += local.numel() * local.element_size()
            locals_.append(local)
            del old
        module._cpu_ngram_lookup = FrozenCpuNgramLookup(locals_, copy_weights=False)
        module._lookup_local_rows = MethodType(_cpu_local_lookup, module)
    model._persistent_extra_parallel_param_ids = persistent_ids
    logger.info("Frozen CPU PLE installed: modules=%s local_bytes=%s", len(modules), moved_bytes)
    return moved_bytes


def refresh_cpu_ngram_lookup(model):
    """Rebind detached lookup views after DCP has restored current parameters."""
    for name, module in _ngram_modules(model):
        lookup = getattr(module, "_cpu_ngram_lookup", None)
        if lookup is None:
            raise ValueError(f"{name}: CPU PLE was not installed before restore")
        weights = []
        for embedding in module.ngram_embedding.values():
            parameter = embedding.weight
            if (
                not isinstance(parameter, DTensor)
                or parameter.placements != (Shard(1), Shard(0))
                or parameter.device.type != "cpu"
                or parameter.requires_grad
                or parameter.is_meta
            ):
                raise ValueError(f"{name}: invalid restored CPU PLE parameter")
            weights.append(parameter.to_local())
        rebound = FrozenCpuNgramLookup(weights, copy_weights=False)
        # Preserve the observer object and timing attachment.
        lookup._weights = rebound._weights
    # Existing IDs for other persistent modules remain valid; these tables are
    # frozen and therefore have no optimizer state or gradient clipping work.
    model._persistent_extra_parallel_param_ids = {
        id(p) for p in model.parameters() if id(p) in getattr(model, "_persistent_extra_parallel_param_ids", set())
    } | {id(e.weight) for _, m in _ngram_modules(model) for e in m.ngram_embedding.values()}
