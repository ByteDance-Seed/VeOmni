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

"""Installation contract for the optional fla_npu runtime used by NPU images."""

from __future__ import annotations

import importlib
import warnings
from importlib import metadata

import pytest
import torch
from packaging.utils import canonicalize_name

from veomni.utils.import_utils import is_torch_npu_available


pytestmark = pytest.mark.skipif(not is_torch_npu_available(), reason="Ascend NPU test")

_SUPPORTED_DISTRIBUTIONS = ("fla-npu", "flash-linear-attention-npu")
_REQUIRED_TORCH_OPS = (
    "npu_recompute_w_u_fwd",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_chunk_fwd_o",
    "npu_chunk_bwd_dv_local",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dqkwg",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
)


def _installed_fla_npu_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for candidate in _SUPPORTED_DISTRIBUTIONS:
        try:
            distribution = metadata.distribution(candidate)
        except metadata.PackageNotFoundError:
            continue

        name = canonicalize_name(distribution.metadata["Name"])
        installed[name] = distribution.version

    return installed


def test_fla_npu_distribution_is_installed() -> None:
    """Require an installed distribution and warn about mixed installations."""
    installed = _installed_fla_npu_distributions()
    assert installed, "No fla_npu distribution is installed. Expected one of: " + ", ".join(_SUPPORTED_DISTRIBUTIONS)
    if len(installed) > 1:
        warnings.warn(
            f"Multiple fla_npu distributions are installed: {installed}. "
            "Uninstall the stale distribution and keep only one fla_npu distribution. "
            "Continuing with the import and operator registration checks.",
            UserWarning,
            stacklevel=2,
        )


def test_fla_npu_import_registers_veomni_required_torch_ops() -> None:
    """The installed distribution must preserve VeOmni's current runtime API."""
    importlib.import_module("torch_npu")
    fla_npu = importlib.import_module("fla_npu")
    assert fla_npu.__name__ == "fla_npu"

    missing = [name for name in _REQUIRED_TORCH_OPS if not hasattr(torch.ops.npu, name)]
    assert not missing, (
        f"fla_npu does not register the torch.ops.npu operators required by VeOmni: {missing}. "
        "Build/install the legacy torch-ops compatibility layer or migrate VeOmni to fla_npu.ops.ascendc."
    )
