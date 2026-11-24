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

from datetime import timedelta

import pytest

from veomni.trainer import base


def _patch_dist(monkeypatch, backend, calls):
    monkeypatch.setattr(base.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(base, "get_dist_comm_backend", lambda: backend)
    monkeypatch.setattr(base.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))


def test_nccl_timeout_is_forwarded_to_process_group(monkeypatch):
    calls = []
    monkeypatch.setenv("NCCL_TIMEOUT", "1800")
    _patch_dist(monkeypatch, "nccl", calls)

    base._init_process_group()

    assert calls == [{"backend": "nccl", "timeout": timedelta(seconds=1800)}]


def test_process_group_keeps_backend_default_without_override(monkeypatch):
    calls = []
    monkeypatch.delenv("NCCL_TIMEOUT", raising=False)
    _patch_dist(monkeypatch, "nccl", calls)

    base._init_process_group()

    assert calls == [{"backend": "nccl"}]


@pytest.mark.parametrize("value", ["0", "-1", "ten minutes"])
def test_nccl_timeout_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("NCCL_TIMEOUT", value)
    _patch_dist(monkeypatch, "nccl", [])

    with pytest.raises(ValueError, match="NCCL_TIMEOUT must be a positive integer"):
        base._init_process_group()


def test_nccl_timeout_does_not_change_other_backends(monkeypatch):
    calls = []
    monkeypatch.setenv("NCCL_TIMEOUT", "invalid-for-nccl")
    _patch_dist(monkeypatch, "hccl", calls)

    base._init_process_group()

    assert calls == [{"backend": "hccl"}]


def test_an_initialized_process_group_is_left_alone(monkeypatch):
    calls = []
    monkeypatch.setenv("NCCL_TIMEOUT", "1800")
    _patch_dist(monkeypatch, "nccl", calls)
    monkeypatch.setattr(base.dist, "is_initialized", lambda: True)

    base._init_process_group()

    assert calls == []
