# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License").

"""Collective preflight for the opt-in full-world PLE file-state contract.

Standard trainers establish this once on every WORLD rank before model build.
The receipt is bound to the model arguments and then explicitly passed to the
optimizer; it is never serialized in a configuration or checkpoint. Direct
users must call establish on all WORLD ranks and pass the returned receipt.
"""

import json
import os
from dataclasses import asdict, dataclass, is_dataclass

import torch.distributed as dist

from ..utils.dist_utils import raise_if_any_rank_failed, require_same_rank_value


@dataclass(frozen=True)
class _Receipt:
    pid: int
    process_group: object
    owner_id: int | None
    optimizer: str
    geometry: str
    save_async: bool
    checkpoint_manager: str


@dataclass(frozen=True)
class _FactoryPermit:
    receipt: _Receipt


_receipt = None


def _json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "__dict__"):
        value = vars(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _requested(optimizer):
    return getattr(optimizer, "type", None) == "qwen38_muon" and (
        getattr(optimizer, "qwen38_ple_moments_world_profile", False)
        or getattr(optimizer, "qwen38_ple_moments_scratch_dir", None) is not None
    )


def _fields(optimizer, accelerator, checkpoint):
    kind = getattr(optimizer, "type", None)
    world_profile = getattr(optimizer, "qwen38_ple_moments_world_profile", False)
    scratch = getattr(optimizer, "qwen38_ple_moments_scratch_dir", None)
    save_async = getattr(checkpoint, "save_async", False)
    manager = getattr(checkpoint, "manager", "dcp")
    if not isinstance(world_profile, bool) or not isinstance(save_async, bool):
        raise TypeError("PLE world profile and save_async must be boolean")
    if scratch is not None and not isinstance(scratch, str):
        raise TypeError("PLE scratch directory must be a string")
    return (kind, world_profile, scratch, save_async, manager, _json_value(optimizer), _json_value(accelerator))


def establish(optimizer, accelerator, checkpoint, *, owner=None):
    """Agree the file profile on every WORLD rank before optimizer branching."""
    global _receipt
    _receipt = None
    error = None
    try:
        requested = bool(_requested(optimizer))
    except Exception as exc:
        requested, error = False, f"{type(exc).__name__}: {exc}"
    if not dist.is_initialized() and requested and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Distributed PLE file preflight requires an initialized WORLD process group")
    payload = (requested, error)
    if dist.is_initialized():
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, payload)
    else:
        values = [payload]
    if any(item[1] is not None for item in values):
        raise ValueError(f"Could not inspect PLE optimizer profile on all ranks: {values}")
    if not any(item[0] for item in values):
        return None

    # Unrelated optimizers may carry custom, non-JSON-serializable options.
    # Only inspect the full contract after at least one rank requests this
    # opt-in profile, while keeping the same collectives on every WORLD rank.
    error = None
    try:
        local = _fields(optimizer, accelerator, checkpoint)
    except Exception as exc:
        local, error = None, f"{type(exc).__name__}: {exc}"
    payload = (local, error)
    if dist.is_initialized():
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, payload)
    else:
        values = [payload]
    if any(item[1] is not None for item in values):
        raise ValueError(f"Could not inspect PLE optimizer profile on all ranks: {values}")
    signatures = [item[0] for item in values]
    if any(sig != signatures[0] for sig in signatures[1:]):
        raise ValueError("PLE file optimizer profile differs across WORLD ranks")
    kind, world_profile, scratch, save_async, manager, opt, geometry = signatures[0]
    if kind != "qwen38_muon" or not world_profile or not scratch or save_async or manager != "dcp":
        raise ValueError(
            "PLE file profile requires Qwen Muon, full-world participation, local scratch and synchronous DCP"
        )
    result = _Receipt(
        os.getpid(),
        dist.group.WORLD if dist.is_initialized() else None,
        None if owner is None else id(owner),
        opt,
        geometry,
        save_async,
        manager,
    )
    if owner is not None:
        _receipt = result
    return result


def _validate_receipt(receipt):
    if not isinstance(receipt, _Receipt) or receipt.pid != os.getpid():
        raise RuntimeError("Invalid PLE file profile receipt")
    if receipt.process_group is None and dist.is_initialized():
        raise RuntimeError("PLE file profile was created before the WORLD group; repeat preflight")
    if receipt.process_group is not None and (
        not dist.is_initialized() or receipt.process_group is not dist.group.WORLD
    ):
        raise RuntimeError("PLE file profile receipt belongs to another WORLD group")


def for_job(owner):
    if _receipt is None:
        return None
    try:
        if getattr(_receipt, "owner_id", None) != id(owner):
            return None
    except Exception:
        # Defer malformed receipt errors to the synchronized validator.
        return _receipt
    # Validation is collective in validate_before_build; raising locally here
    # could leave the other profile ranks waiting in their first collective.
    return _receipt


def validate_before_build(optimizer, accelerator, checkpoint, receipt, *, model_name="base"):
    """Validate the bound config; file-profile WORLD ranks must enter in order.

    Ordinary non-profile optimizer construction remains a local operation.
    """
    local, error = None, None
    try:
        requested = bool(_requested(optimizer))
    except Exception as exc:
        if receipt is None:
            # Without an opt-in receipt this may be an ordinary, rank-local
            # optimizer call. Do not introduce a WORLD collective there.
            raise
        requested, error = False, exc
    if receipt is None and not requested and error is None:
        return
    try:
        if error is not None:
            raise error
        if receipt is None:
            if not dist.is_initialized():
                return
            raise RuntimeError("Distributed PLE file profile requires a WORLD preflight on all ranks")
        _validate_receipt(receipt)
        local = _fields(optimizer, accelerator, checkpoint)
        if model_name != "base":
            raise ValueError("PLE file profile currently supports the standard base model only")
        if (local[5], local[6], local[3], local[4]) != (
            receipt.optimizer,
            receipt.geometry,
            receipt.save_async,
            receipt.checkpoint_manager,
        ):
            raise ValueError("PLE file profile configuration changed after preflight")
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating the PLE optimizer job profile")
    require_same_rank_value(local, "PLE optimizer job profile")


def validate_constructor(scratch, world_profile, receipt, permit):
    """Validate locally: the factory coordinates all leaf construction errors."""
    if not (scratch is not None or world_profile):
        return
    if not dist.is_initialized():
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Distributed PLE file optimizer requires an initialized WORLD process group")
        return
    _validate_receipt(receipt)
    if not isinstance(permit, _FactoryPermit) or permit.receipt is not receipt:
        raise RuntimeError("Distributed PLE file optimizers must use the certified optimizer factory")
    expected = json.loads(receipt.optimizer)
    if scratch != expected.get("qwen38_ple_moments_scratch_dir") or world_profile is not True:
        raise ValueError("PLE file optimizer differs from its job profile")


def validate_build_config(kind, optimizer, receipt):
    """Guard a file-profile factory collectively after all-WORLD preflight."""
    if not dist.is_initialized():
        return
    error = None
    try:
        requested = bool(_requested(optimizer))
    except Exception as exc:
        if receipt is None:
            raise
        requested, error = False, exc
    if receipt is None and not requested and error is None:
        return
    try:
        if error is not None:
            raise error
        if receipt is None:
            raise RuntimeError("Distributed PLE file optimizer factory requires an all-rank WORLD preflight")
        _validate_receipt(receipt)
        if kind != "qwen38_muon" or _json_value(optimizer) != receipt.optimizer:
            raise ValueError("Optimizer factory configuration changed after PLE world preflight")
    except Exception as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating the PLE optimizer factory")
