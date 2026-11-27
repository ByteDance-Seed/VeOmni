# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Hardware checks shared by the low-precision rows and their quantizers."""

from __future__ import annotations

from functools import cache

import torch
from torch import nn

from ..platform import NVIDIA_SM90_PLUS


_TORCHAO_ROW = "torchao"
_TORCHAO_FIELDS = (
    ("linear", "linear_implementation"),
    ("gated_mlp", "gated_mlp_implementation"),
)


def _torchao_pair(kwargs: dict) -> tuple[str, bool]:
    dtype = kwargs.get("dtype", "nvfp4")
    emulated = kwargs.get("emulated")
    return (dtype, (dtype != "fp8") if emulated is None else bool(emulated))


def _torchao_configs_for_selection(value) -> list[tuple[str, bool]]:
    """Collect ``(dtype, emulated)`` pairs reachable for ``torchao`` under one selection.

    Each override's ``site`` is a regex; for a concrete site, only the matching
    rules apply in order. Without concrete site names, track every reachable
    ``(is_torchao, dtype, emulated)`` state across ordered subsets of overrides
    so chained overrides (e.g. one rule setting ``name='torchao'`` and a later
    rule setting ``kwargs={'dtype': 'fp8'}``) are accounted for.
    """
    base_name = getattr(value, "name", value)
    base_kwargs = dict(getattr(value, "kwargs", {}) or {})
    states: set[tuple[str | None, str, bool | None]] = {
        (base_name, base_kwargs.get("dtype", "nvfp4"), base_kwargs.get("emulated"))
    }
    for rule in getattr(value, "overrides", ()) or ():
        rule_kwargs = rule.kwargs or {}
        next_states = set(states)
        for name, dtype, emulated in states:
            if rule.name is not None and name is not None and rule.name != name:
                next_states.add((rule.name, rule_kwargs.get("dtype", "nvfp4"), rule_kwargs.get("emulated")))
            else:
                next_states.add(
                    (
                        rule.name if rule.name is not None else name,
                        rule_kwargs.get("dtype", dtype),
                        rule_kwargs.get("emulated", emulated),
                    )
                )
        states = next_states
    return [
        (dtype, (dtype != "fp8") if emulated is None else bool(emulated))
        for name, dtype, emulated in states
        if name == _TORCHAO_ROW
    ]


def _bound_torchao_configs(model) -> tuple[bool, list[tuple[str, bool]]]:
    """Inspect concrete ``VeomniOp`` handles bound on ``model`` for ``linear`` / ``gated_mlp``."""
    if not isinstance(model, nn.Module):
        return False, []
    from ..registry import VeomniOp

    tracked_ops = {op for op, _ in _TORCHAO_FIELDS}
    has_bound_ops = False
    found: list[tuple[str, bool]] = []
    for module in model.modules():
        for attr in vars(module).values():
            if isinstance(attr, VeomniOp):
                has_bound_ops = True
                if attr.op in tracked_ops and attr.impl == _TORCHAO_ROW:
                    found.append(_torchao_pair(attr.kwargs))
    return has_bound_ops, found


def _iter_torchao_configs(config) -> list[tuple[str, bool]]:
    """Collect ``(dtype, emulated)`` pairs reachable for ``torchao`` across ``config`` and sub-configs."""
    found: list[tuple[str, bool]] = []
    if config is None:
        return found

    ops = getattr(config, "ops_config", None)
    if ops is not None:
        for _op, field in _TORCHAO_FIELDS:
            found.extend(_torchao_configs_for_selection(getattr(ops, field, None)))

    for name in getattr(config, "sub_configs", {}) or {}:
        child = getattr(config, name, None)
        if child is not None:
            found.extend(_iter_torchao_configs(child))
    return found


def _model_torchao_configs(model: nn.Module) -> list[tuple[str, bool]]:
    """Collect ``torchao`` ``(dtype, emulated)`` pairs for ``model``.

    On a concrete model that binds ``VeomniOp`` handles, inspect only those
    resolved handles so unmatched overrides or op families the model never
    binds (such as ``linear_implementation`` on Gemma3) do not falsely fail
    hardware checks. Fall back to reachable-state config inspection when no
    handles are bound on ``model``.
    """
    has_bound_ops, found = _bound_torchao_configs(model)
    if has_bound_ops:
        return found
    return _iter_torchao_configs(getattr(model, "config", None))


def _native_bf16_qat_selections(model: nn.Module) -> set[str]:
    """Selected torchao rows on ``model``'s bound handles or config tree, child configs included."""
    return {_TORCHAO_ROW} if _model_torchao_configs(model) else set()


def reject_qat_without_native_bf16(model: nn.Module) -> None:
    """Refuse unsupported CUDA hardware for selected torchao low-precision rows.

    * Native FP8 (``dtype="fp8"``, ``emulated=False``) requires NVIDIA CUDA SM89+
      (Ada, Hopper, Blackwell; ROCm is unsupported).
    * Emulated FP4/FP8 rows require a CUDA device with native BF16 support
      (SM80 or newer) during distributed training.

    A training-path precondition rather than a row ``requirement``: emulated rows
    also run on CPU for op-level tests, so non-CUDA hosts are skipped here.
    """
    configs = _model_torchao_configs(model)
    if not configs:
        return
    from ...utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability

    if not IS_CUDA_AVAILABLE:
        return

    cc = get_gpu_compute_capability()
    if any(dtype == "fp8" and not emulated for dtype, emulated in configs):
        if torch.version.hip or cc < 89:
            raise RuntimeError(
                "torchao native float8 (dtype='fp8', emulated=False) requires an NVIDIA CUDA SM89 or newer "
                "device; this host cannot run native float8 GEMMs."
            )
    if cc < 80:
        raise RuntimeError(
            "torchao low-precision rows require a CUDA device with native BF16 support (SM80 or newer); "
            "this host reports an older compute capability."
        )


@cache
def require_tilelang_sm90() -> None:
    """Reject unsupported hardware before importing the TileLang quantizers."""
    if not NVIDIA_SM90_PLUS.matches():
        raise RuntimeError("QAT TileLang quantizers require an SM90 or later NVIDIA CUDA GPU")
