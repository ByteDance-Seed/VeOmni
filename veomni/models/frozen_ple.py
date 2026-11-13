"""Frozen Qwen4-Exp PLE tables: optimizer exclusion and dtype conversion.

Same rule as verl's ``verl.utils.veomni.frozen_ple``: only parameters named
``ple.ple_embedding.ngram_embedding.shard_<n>.weight`` are frozen. The forward
lookup still reads them.
"""

import re

import torch


_PLE_TABLE = re.compile(r"(?:^|\.)ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")


def is_ple_table(name: str) -> bool:
    """Return whether a parameter name identifies a PLE n-gram shard weight."""
    return _PLE_TABLE.search(name) is not None


def prepare_ple_storage(model: torch.nn.Module, dtype: str | None, expected_device: str | None = None) -> None:
    """Apply frozen PLE dtype and validate shard placement, preserving parameter identity."""
    if dtype not in (None, "float32", "bfloat16"):
        raise ValueError(f"Unsupported frozen PLE dtype: {dtype}")
    target = getattr(torch, dtype) if dtype is not None else None
    for name, param in model.named_parameters():
        if not is_ple_table(name):
            continue
        local = param.to_local() if hasattr(param, "to_local") else param
        if expected_device is not None and local.device.type != expected_device:
            raise RuntimeError(f"PLE table {name} is on {local.device}, expected {expected_device}")
        if target is not None and param.dtype != target:
            # DTensor .data assignment changes wrapper dtype but leaves its
            # local shard unchanged. Swap the complete tensor implementation.
            converted = torch.nn.Parameter(param.to(dtype=target), requires_grad=param.requires_grad)
            torch.utils.swap_tensors(param, converted)


def freeze_ple_embeddings(model: torch.nn.Module) -> int:
    """Freeze Qwen4-Exp PLE tables before parallelization and return their count."""
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen4_exp":
        raise ValueError("freeze_ple_embeddings currently supports only qwen4_exp")

    count = 0
    for name, param in model.named_parameters():
        if not is_ple_table(name):
            continue
        param.requires_grad_(False)
        param.grad = None
        count += 1
    if not count:
        raise ValueError("freeze_ple_embeddings requested but no PLE table parameters were found")
    return count
