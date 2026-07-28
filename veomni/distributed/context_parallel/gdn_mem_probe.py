# Copyright (c) 2025 VeOmni Authors.
"""Layered GDN/FA/HBM memory probe for CP A/B contrasts.

Enable with ``VEOMNI_GDN_MEM_PROBE=1``. Rank0 emits one JSON line per sample:

``AI4SE_MEM_LAYER {...}``

Buckets (contract for 256K serial vs gather_full):
- ``gdn_act``: live GDN activation bytes (q/k/v/g/beta/mixed_qkv/core_out/state)
- ``hbm_alloc`` / ``hbm_reserved``: device allocator snapshot (NPU/CUDA)
- ``seq_tokens``: sequence length of the measured tensor (S_local or S_full)

INV-1 evidence needs **same meter, paired runs**: compare ``gdn_act`` serial≪gather_full
even when whole-device peak is FA-dominated.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import torch

_ENABLED: Optional[bool] = None
_STEP: int = 0
_IMPL: str = ""
_SEEN: Dict[str, int] = {}


def mem_probe_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("VEOMNI_GDN_MEM_PROBE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return bool(_ENABLED)


def mem_probe_set_context(*, impl: str, step: Optional[int] = None) -> None:
    global _IMPL, _STEP
    if impl:
        _IMPL = str(impl)
    if step is not None:
        _STEP = int(step)


def _device_of(t: torch.Tensor) -> torch.device:
    return t.device if isinstance(t, torch.Tensor) else torch.device("cpu")


def _alloc_stats(device: torch.device) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "hbm_alloc_bytes": None,
        "hbm_reserved_bytes": None,
        "hbm_max_alloc_bytes": None,
    }
    if device.type == "cpu":
        return out
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            out["hbm_alloc_bytes"] = int(torch.npu.memory_allocated(device))
            out["hbm_reserved_bytes"] = int(torch.npu.memory_reserved(device))
            out["hbm_max_alloc_bytes"] = int(torch.npu.max_memory_allocated(device))
        elif device.type == "cuda":
            out["hbm_alloc_bytes"] = int(torch.cuda.memory_allocated(device))
            out["hbm_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
            out["hbm_max_alloc_bytes"] = int(torch.cuda.max_memory_allocated(device))
    except Exception:
        pass
    return out


def tensor_nbytes(*tensors: Any) -> int:
    total = 0
    for t in tensors:
        if t is None or not isinstance(t, torch.Tensor):
            continue
        total += int(t.numel()) * int(t.element_size())
    return total


def emit_mem_layer(
    tag: str,
    *,
    tensors: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    layer_idx: Optional[int] = None,
    once_per_step: bool = True,
) -> None:
    """Emit a single probe sample. Rank>0 silent. Optional once-per-(step,tag,layer)."""
    if not mem_probe_enabled():
        return
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        rank = 0
    if rank != 0:
        return

    key = f"{_STEP}:{tag}:{layer_idx}"
    if once_per_step:
        # Keep first few layers + a late layer for variance; always allow layer 0/1/2.
        if layer_idx is not None and layer_idx not in (0, 1, 2, 8, 16, 24, 32):
            return
        if key in _SEEN:
            return
        _SEEN[key] = 1

    tensors = tensors or {}
    nbytes_by = {k: tensor_nbytes(v) for k, v in tensors.items() if v is not None}
    total = int(sum(nbytes_by.values()))
    # Prefer a primary activation tensor for seq_tokens
    seq_tokens = None
    primary = None
    for name in ("mixed_qkv", "query", "core_attn_out", "value"):
        t = tensors.get(name)
        if isinstance(t, torch.Tensor) and t.ndim >= 2:
            primary = t
            # assume [B, S, ...]
            seq_tokens = int(t.shape[1])
            break
    device = _device_of(primary) if primary is not None else torch.device("cpu")
    stats = _alloc_stats(device)
    payload = {
        "schema": "ai4se.mem_layer/v1",
        "tag": tag,
        "impl": _IMPL or os.environ.get("VEOMNI_GDN_CP_IMPL", ""),
        "step": _STEP,
        "layer_idx": layer_idx,
        "gdn_act_bytes": total,
        "gdn_act_by_tensor": nbytes_by,
        "seq_tokens": seq_tokens,
        "ts": time.time(),
        **stats,
    }
    if extra:
        payload.update(extra)
    print("AI4SE_MEM_LAYER " + json.dumps(payload, sort_keys=True), flush=True)


def reset_mem_probe_step(step: int) -> None:
    global _STEP, _SEEN
    _STEP = int(step)
    _SEEN = {}
