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

"""The accelerator's compute-unit count, asked the way each backend reports it."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from veomni.utils.device import get_compute_units


def _accelerator(device_type):
    """A torch.accelerator stand-in that reports one accelerator type."""
    return SimpleNamespace(
        is_available=lambda: device_type is not None,
        current_accelerator=lambda: SimpleNamespace(type=device_type),
    )


def test_a_cuda_host_reads_the_sm_count():
    props = SimpleNamespace(multi_processor_count=132)
    with (
        patch.object(torch, "accelerator", _accelerator("cuda")),
        patch.object(torch.cuda, "get_device_properties", lambda index: props),
    ):
        assert get_compute_units() == 132


def test_an_npu_host_reads_the_same_counter_as_cuda():
    """An NPU publishes multi_processor_count too; it is not a CPU-only host.

    The match fell through to the CPU branch for ``npu``, so the persistent
    Triton matmul in ``veomni/ops/batch_invariant/triton.py`` sized its grid to
    ``torch.get_num_threads()`` (256 on the test host) instead of the 40 AI
    cores the device has.
    """
    props = SimpleNamespace(multi_processor_count=40)
    npu = SimpleNamespace(get_device_properties=lambda index: props)
    with (
        patch.object(torch, "accelerator", _accelerator("npu")),
        patch.object(torch, "npu", npu, create=True),
    ):
        assert get_compute_units() == 40


def test_an_xpu_and_an_mlu_host_keep_their_own_counters():
    xpu_props = SimpleNamespace(max_compute_units=96)
    mlu_props = SimpleNamespace(multi_processor_count=64)
    with (
        patch.object(torch, "accelerator", _accelerator("xpu")),
        patch.object(torch, "xpu", SimpleNamespace(get_device_properties=lambda index: xpu_props), create=True),
    ):
        assert get_compute_units() == 96
    with (
        patch.object(torch, "accelerator", _accelerator("mlu")),
        patch.object(torch, "mlu", SimpleNamespace(get_device_properties=lambda index: mlu_props), create=True),
    ):
        assert get_compute_units() == 64


def test_a_cpu_only_host_reports_its_threads():
    with patch.object(torch, "accelerator", _accelerator(None)):
        assert get_compute_units() == torch.get_num_threads()
