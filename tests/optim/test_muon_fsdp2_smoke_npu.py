# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""NPU entry points for the FSDP2 + ExtraParallel Muon smoke coverage.

The original GPU suite remains unchanged. The worker below reuses its complete
distributed assertions and rewrites only the fused MoE backend selected by that
suite from ``fused_triton`` to ``fused_npu``.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.utils.device import IS_NPU_AVAILABLE

from ..tools.launch_utils import find_free_port
from . import test_muon_fsdp2_smoke as _base


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Muon FSDP2 NPU smoke tests require torch_npu")


def _distributed_smoke_npu(use_zero_comm: bool) -> None:
    original_init = OpsImplementationConfig.__init__
    rewritten = 0

    def _npu_init(self, *args, **kwargs):
        nonlocal rewritten
        if kwargs.get("moe_implementation") == "fused_triton":
            kwargs["moe_implementation"] = "fused_npu"
            rewritten += 1
        original_init(self, *args, **kwargs)

    with patch.object(OpsImplementationConfig, "__init__", _npu_init):
        _base._distributed_smoke(use_zero_comm=use_zero_comm)

    assert rewritten == 1, f"expected to rewrite one fused MoE config, rewrote {rewritten}"


def _torchrun_cmd(use_zero_comm: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=4",
        f"--master_port={find_free_port()}",
        "--module",
        "tests.optim.test_muon_fsdp2_smoke_npu",
        f"--zero-comm={int(use_zero_comm)}",
    ]


@pytest.mark.skipif(not _base._has_devices(4), reason="device_count should be >= 4")
@pytest.mark.parametrize("use_zero_comm", [False, True])
def test_muon_fsdp2_smoke_npu(use_zero_comm):
    result = subprocess.run(_torchrun_cmd(use_zero_comm), check=True)
    assert result.returncode == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-comm", type=int, default=0)
    args = parser.parse_args()
    _distributed_smoke_npu(use_zero_comm=bool(args.zero_comm))
