"""Frozen PLE and Engram lookup tables: optimizer exclusion, dtype conversion, and export filtering.

``is_ple_table``, ``freeze_ple_embeddings``, and ``prepare_ple_storage`` manage
Qwen4-Exp ``ple.ple_embedding.ngram_embedding.shard_<n>.weight`` parameters
during training. ``is_engram_or_ple_table`` and
``collect_frozen_engram_state_dict_keys`` identify frozen PLE and Engram lookup
tables (and companion scale tensors) so callers can omit them during partial
weight export without unsharding ``DTensor`` shards.
"""

import re
from collections.abc import Iterable

import torch


_PLE_TABLE = re.compile(r"(?:^|\.)ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")

_ENGRAM_OR_PLE_TABLE_SCOPES = (
    re.compile(
        r"^(?P<scope>(?:.+\.)?ngram_embedding(?:\.shard_\d+)?)\.(?:weight|weight_scale|scale|weight_scale_inv)$"
    ),
    re.compile(
        r"^(?P<scope>(?:.+\.)?engram(?:\.\d+)?\.(?:embed|embed_tokens))\.(?:weight|scale|weight_scale|weight_scale_inv)$"
    ),
    re.compile(
        r"^(?P<scope>(?:.+\.)?(?:engram|ngram_embedding|ngram_embeddings)\.tables\.\d+)\.(?:weight|scale|weight_scale|weight_scale_inv)$"
    ),
)


def _strip_ddp_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


def _engram_table_scope(name: str) -> str | None:
    for pattern in _ENGRAM_OR_PLE_TABLE_SCOPES:
        match = pattern.match(name)
        if match is not None:
            return match.group("scope")
    return None


def is_ple_table(name: str) -> bool:
    """Return whether a parameter name identifies a PLE n-gram shard weight."""
    return _PLE_TABLE.search(name) is not None


def is_engram_or_ple_table(name: str) -> bool:
    """Return whether ``name`` identifies an Engram or PLE n-gram lookup table tensor/scale."""
    return _engram_table_scope(_strip_ddp_prefix(name)) is not None


def collect_frozen_engram_state_dict_keys(
    model: torch.nn.Module,
    state_dict_keys: Iterable[str] | None = None,
) -> set[str]:
    """Return ``state_dict`` keys belonging to frozen Engram / PLE lookup tables.

    Inspects ``model.named_parameters()`` rather than ``model.state_dict()``
    values because ``state_dict()`` detaches tensors (setting ``requires_grad``
    to ``False`` unconditionally) and so callers can skip frozen ``DTensor``
    shards *before* calling ``full_tensor()``. Companion scale buffers in
    ``state_dict_keys`` under the same frozen table scope are included as well.
    """
    frozen_keys: set[str] = set()
    frozen_scopes: set[str] = set()
    trainable_scopes: set[str] = set()

    for raw_name, param in model.named_parameters():
        clean_name = _strip_ddp_prefix(raw_name)
        scope = _engram_table_scope(clean_name)
        if scope is None:
            continue
        if param.requires_grad:
            trainable_scopes.add(scope)
        else:
            frozen_keys.add(raw_name)
            frozen_keys.add(clean_name)
            frozen_scopes.add(scope)

    pure_frozen_scopes = frozen_scopes - trainable_scopes
    if state_dict_keys is not None and pure_frozen_scopes:
        for key in state_dict_keys:
            clean_key = _strip_ddp_prefix(key)
            if clean_key in frozen_keys:
                frozen_keys.add(key)
                continue
            scope = _engram_table_scope(clean_key)
            if scope is not None and scope in pure_frozen_scopes:
                frozen_keys.add(key)
                frozen_keys.add(clean_key)

    return frozen_keys


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
            if hasattr(param, "spec_info"):
                converted.spec_info = param.spec_info
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
