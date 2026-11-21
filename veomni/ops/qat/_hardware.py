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


def _iter_torchao_configs(config) -> list[tuple[str, bool]]:
    """Collect ``(dtype, emulated)`` pairs selected for the ``torchao`` row across ``config`` and sub-configs."""
    found: list[tuple[str, bool]] = []
    if config is None:
        return found

    ops = getattr(config, "ops_config", None)
    if ops is not None:
        for field in ("linear_implementation", "gated_mlp_implementation"):
            value = getattr(ops, field, None)
            base_name = getattr(value, "name", value)
            base_kwargs = dict(getattr(value, "kwargs", {}) or {})
            if base_name == _TORCHAO_ROW:
                dtype = base_kwargs.get("dtype", "nvfp4")
                emulated = base_kwargs.get("emulated")
                found.append((dtype, (dtype != "fp8") if emulated is None else bool(emulated)))
            for rule in getattr(value, "overrides", ()) or ():
                rule_name = rule.name if rule.name is not None else base_name
                if rule_name != _TORCHAO_ROW:
                    continue
                merged = {**base_kwargs, **(rule.kwargs or {})}
                dtype = merged.get("dtype", "nvfp4")
                emulated = merged.get("emulated")
                found.append((dtype, (dtype != "fp8") if emulated is None else bool(emulated)))

    for name in getattr(config, "sub_configs", {}) or {}:
        child = getattr(config, name, None)
        if child is not None:
            found.extend(_iter_torchao_configs(child))
    return found


def _native_bf16_qat_selections(model: nn.Module) -> set[str]:
    """Selected torchao rows on ``model``'s config tree, child configs included."""
    return {_TORCHAO_ROW} if _iter_torchao_configs(getattr(model, "config", None)) else set()


def reject_qat_without_native_bf16(model: nn.Module) -> None:
    """Refuse unsupported CUDA hardware for selected torchao low-precision rows.

    * Native FP8 (``dtype="fp8"``, ``emulated=False``) requires NVIDIA CUDA SM89+
      (Ada, Hopper, Blackwell; ROCm is unsupported).
    * Emulated FP4/FP8 rows require a CUDA device with native BF16 support
      (SM80 or newer) during distributed training.

    A training-path precondition rather than a row ``requirement``: emulated rows
    also run on CPU for op-level tests, so non-CUDA hosts are skipped here.
    """
    configs = _iter_torchao_configs(getattr(model, "config", None))
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
