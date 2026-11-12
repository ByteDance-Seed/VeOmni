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

"""models weight I/O: empty init and save/load.

Model-level ``weights_path`` loading (dense, merged MoE, VL, Omni) lives in
``tests/models/base/test_low_precision_oracle.py``. This file only covers the
tensor I/O primitives.
"""

from __future__ import annotations

import errno
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from veomni.models.checkpoint import weights
from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights, save_model_weights


def test_init_empty_weights_puts_params_on_meta():
    with init_empty_weights():
        module = torch.nn.Linear(3, 3)
    assert module.weight.is_meta
    assert module.bias.is_meta


def test_save_and_load_roundtrip(tmp_path):
    src = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        src.weight.fill_(1.5)
        src.bias.fill_(0.25)

    save_model_weights(str(tmp_path), src.state_dict(), global_rank=0, save_dtype="float32")

    dst = torch.nn.Linear(4, 4, bias=True)
    dst.config = SimpleNamespace()
    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.zero_()
    load_model_weights(dst, str(tmp_path), init_device="cpu")

    assert torch.equal(dst.weight, src.weight)
    assert torch.equal(dst.bias, src.bias)


def test_safetensors_enosys_stages_one_shard_without_changing_bytes(tmp_path, monkeypatch):
    tensors = {
        "fp32": torch.tensor([0.0, -0.0, float("nan"), 0.12345679], dtype=torch.float32),
        "bf16": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
        "integer": torch.tensor([0, 2, -1], dtype=torch.int64),
        "empty": torch.empty(0),
    }
    expected, target = tmp_path / "expected.safetensors", tmp_path / "model.safetensors"
    save_file(tensors, expected, metadata={"format": "pt"})
    calls = []

    def inject_enosys(data, path, metadata):
        calls.append(path)
        if len(calls) == 1:
            raise SafetensorError("Error while serializing: I/O error: Function not implemented (os error 38)")
        save_file(data, path, metadata=metadata)

    monkeypatch.setattr(weights, "save_file", inject_enosys)
    weights._save_state_dict(tensors, target, safe_serialization=True, legacy_fuse_fallback=True)

    assert len(calls) == 2
    assert target.read_bytes() == expected.read_bytes()
    assert target.stat().st_mode & 0o777 == expected.stat().st_mode & 0o777
    with safe_open(target, framework="pt", device="cpu") as saved:
        assert saved.metadata() == {"format": "pt"}
        for name, tensor in tensors.items():
            assert torch.equal(saved.get_tensor(name).view(torch.uint8), tensor.view(torch.uint8))
    assert not list(tmp_path.glob(".veomni-safetensors-*"))


def test_sharded_hf_export_recovers_from_enosys(tmp_path, monkeypatch):
    tensors = {
        "weight": torch.arange(8, dtype=torch.float32),
        "bias": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
    }
    calls = []

    def inject_enosys(data, path, metadata):
        calls.append(path)
        if str(path).startswith(str(tmp_path)):
            raise SafetensorError("Error while serializing: I/O error: Function not implemented (os error 38)")
        save_file(data, path, metadata=metadata)

    monkeypatch.setattr(weights, "save_file", inject_enosys)
    save_model_weights(str(tmp_path), tensors, save_dtype=None, shard_size=16, legacy_fuse_fallback=True)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 2
    for name, filename in index["weight_map"].items():
        with safe_open(tmp_path / filename, framework="pt", device="cpu") as saved:
            assert torch.equal(saved.get_tensor(name).view(torch.uint8), tensors[name].view(torch.uint8))
    assert len(calls) == 4
    assert not list(tmp_path.glob(".veomni-safetensors-*"))


@pytest.mark.parametrize("failure", ["other_error", "copy_error", "rename_error"])
@pytest.mark.parametrize("target_exists", [False, True])
def test_safetensors_fallback_failure_preserves_target(tmp_path, monkeypatch, failure, target_exists):
    target = tmp_path / "model.safetensors"
    if target_exists:
        target.write_bytes(b"old valid file")
    calls = 0

    def inject_failure(data, path, metadata):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Older safetensors versions can truncate their target before failing.
            with open(path, "wb") as output:
                output.write(b"partial old serializer output")
            if failure == "other_error":
                raise SafetensorError("Error while serializing: I/O error: Permission denied (os error 13)")
            raise SafetensorError("Error while serializing: I/O error: Function not implemented (os error 38)")
        save_file(data, path, metadata=metadata)

    monkeypatch.setattr(weights, "save_file", inject_failure)
    if failure == "copy_error":

        def fail_copy(source, destination, **kwargs):
            destination.write(b"partial data")
            raise OSError("injected copy failure")

        monkeypatch.setattr(weights.shutil, "copyfileobj", fail_copy)
    elif failure == "rename_error":

        def fail_rename(*args, **kwargs):
            raise OSError("injected rename failure")

        monkeypatch.setattr(weights.os, "replace", fail_rename)

    expected_error = SafetensorError if failure == "other_error" else OSError
    with pytest.raises(expected_error):
        weights._save_state_dict({"weight": torch.ones(4)}, target, safe_serialization=True, legacy_fuse_fallback=True)
    if target_exists:
        assert target.read_bytes() == b"old valid file"
    else:
        assert not target.exists()
    assert calls == (1 if failure == "other_error" else 2)
    assert not list(tmp_path.glob(".veomni-safetensors-*"))


def test_safetensors_failed_cleanup_does_not_mask_publish_error(tmp_path, monkeypatch):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"previous")
    real_rmtree = weights.shutil.rmtree
    calls = 0

    def inject_enosys(data, path, metadata):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SafetensorError("Error while serializing: I/O error: Function not implemented (os error 38)")
        save_file(data, path, metadata=metadata)

    def fail_publish(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected publish failure")

    def fail_target_cleanup(path, *args, **kwargs):
        if str(path).startswith(str(tmp_path / ".veomni-safetensors-")):
            raise OSError(errno.ENOSYS, "injected cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(weights, "save_file", inject_enosys)
    monkeypatch.setattr(weights.os, "replace", fail_publish)
    monkeypatch.setattr(weights.shutil, "rmtree", fail_target_cleanup)
    with pytest.raises(OSError) as exc:
        weights._save_state_dict({"weight": torch.ones(4)}, target, safe_serialization=True, legacy_fuse_fallback=True)
    assert exc.value.errno == errno.ENOSPC
    assert target.read_bytes() == b"previous"


def test_safetensors_success_ignores_stage_cleanup_failure(tmp_path, monkeypatch):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"previous")
    real_rmtree = weights.shutil.rmtree

    def fail_target_cleanup(path, *args, **kwargs):
        if str(path).startswith(str(tmp_path / ".veomni-safetensors-")):
            raise OSError(errno.ENOSYS, "injected cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(weights.shutil, "rmtree", fail_target_cleanup)
    weights._save_state_dict({"weight": torch.ones(4)}, target, safe_serialization=True, legacy_fuse_fallback=True)
    with safe_open(target, framework="pt", device="cpu") as saved:
        assert torch.equal(saved.get_tensor("weight"), torch.ones(4))


def test_safetensors_new_target_publishes_after_save(tmp_path, monkeypatch):
    target = tmp_path / "model.safetensors"
    real_replace = weights.os.replace
    calls = []

    def record_replace(source, destination):
        calls.append((source, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(weights.os, "replace", record_replace)
    weights._save_state_dict({"weight": torch.ones(4)}, target, safe_serialization=True, legacy_fuse_fallback=True)
    # This counts VeOmni's publish operation, not renames internal to safetensors.
    assert len(calls) == 1
    assert calls[0][1] == target
    with safe_open(target, framework="pt", device="cpu") as saved:
        assert torch.equal(saved.get_tensor("weight"), torch.ones(4))


def test_safetensors_staging_is_not_enabled_for_other_callers(tmp_path, monkeypatch):
    calls = 0

    def inject_enosys(data, path, metadata):
        nonlocal calls
        calls += 1
        raise SafetensorError("Error while serializing: I/O error: Function not implemented (os error 38)")

    monkeypatch.setattr(weights, "save_file", inject_enosys)
    with pytest.raises(SafetensorError, match="os error 38"):
        weights._save_state_dict({"weight": torch.ones(1)}, tmp_path / "other.safetensors", True)
    assert calls == 1
