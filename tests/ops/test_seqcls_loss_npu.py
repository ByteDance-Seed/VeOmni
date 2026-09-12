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

"""NPU entry points for the device-agnostic sequence-classification losses.

The Liger fused-linear case in the original suite is a GPU-only dependency and
is intentionally not re-exported here. The remaining assertions are reused
without changing their source.
"""

import pytest

from tests.ops import test_seqcls_loss as _base
from veomni.utils.device import IS_NPU_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="Sequence-classification NPU tests require torch_npu")


def test_seqcls_loss_logits_path_manual_handcalc_npu(monkeypatch):
    _base.test_seqcls_loss_logits_path_manual_handcalc(monkeypatch)


def test_seqcls_loss_hidden_states_weights_path_build_logits_and_loss_npu(monkeypatch):
    _base.test_seqcls_loss_hidden_states_weights_path_build_logits_and_loss(monkeypatch)


def test_seqcls_loss_sp_enabled_calls_reduce_with_correct_num_valid_tokens_npu(monkeypatch):
    _base.test_seqcls_loss_sp_enabled_calls_reduce_with_correct_num_valid_tokens(monkeypatch)


def test_seqcls_loss_assertions_npu(monkeypatch):
    _base.test_seqcls_loss_assertions(monkeypatch)


def test_reduce_loss_no_nan_when_sp_group_all_padding_npu():
    _base.test_reduce_loss_no_nan_when_sp_group_all_padding()
