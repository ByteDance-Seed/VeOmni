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

"""Regression tests for chunk loss under activation-offload hooks."""

from functools import partial
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks

import veomni.ops.kernels.cross_entropy as cross_entropy_module
import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_loss_module
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.offloading import custom_save_on_cpu
from veomni.ops.config.singleton import get_ops_config, set_ops_config
from veomni.ops.kernels.cross_entropy.chunk_loss import ChunkLoss


def _loss_forward(hidden_states, weight, bias, labels, ignore_index):
    assert bias is None
    logits = F.linear(hidden_states.reshape(-1, hidden_states.size(-1)), weight)
    loss = F.cross_entropy(logits, labels.reshape(-1), ignore_index=ignore_index, reduction="sum")
    return loss, logits


def _chunked_loss(hidden_states, weight, labels, chunk_size, ignore_index=-100):
    label_chunks = torch.split(labels, chunk_size, dim=1)
    loss_kwargs_chunks = [{"labels": chunk, "ignore_index": ignore_index} for chunk in label_chunks]
    return ChunkLoss.apply(hidden_states, weight, None, _loss_forward, loss_kwargs_chunks, chunk_size)


def _reference_loss(hidden_states, weight, labels, ignore_index=-100):
    logits = F.linear(hidden_states.reshape(-1, hidden_states.size(-1)), weight)
    return F.cross_entropy(logits, labels.reshape(-1), ignore_index=ignore_index, reduction="sum")


def test_chunk_loss_supports_saved_tensor_hooks_and_preserves_gradients():
    """Activation-offload hooks run and first-order loss/gradients stay equal."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, 4, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 7, (2, 5))
    labels[0, 1] = -100
    labels[1, 4] = -100
    upstream_scale = 0.37

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = _reference_loss(reference_hidden, reference_weight, labels)
    (reference * upstream_scale).backward()

    hook_calls = {"pack": 0, "unpack": 0}

    def pack(tensor):
        hook_calls["pack"] += 1
        return tensor

    def unpack(tensor):
        hook_calls["unpack"] += 1
        return tensor

    # chunk_size=2 exercises multiple chunks and a one-token tail chunk.
    with saved_tensors_hooks(pack, unpack):
        actual = _chunked_loss(hidden, weight, labels, chunk_size=2)
        (actual * upstream_scale).backward()

    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=1e-6, atol=1e-6)
    # Only the two long-lived tensors saved by the custom Function reach the
    # outer offload hook; the immediately differentiated CE graph is scoped to
    # the inner identity hook.
    assert hook_calls == {"pack": 2, "unpack": 2}


def test_full_chunk_loss_runs_under_activation_offload_hook(monkeypatch):
    """Exercise the production causal wrapper with the actual offload hook."""

    class _FakeParallelState:
        sp_enabled = False

    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: _FakeParallelState())
    torch.manual_seed(1)
    hidden = torch.randn(2, 6, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, 4, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 7, (2, 6))
    labels[0, 2] = -100
    scale = 0.37

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference, _ = chunk_loss_module.chunk_loss_function(
        reference_hidden, reference_weight, labels, chunk_size=2, vocab_size=7
    )
    (reference * scale).backward()

    with custom_save_on_cpu(gpu_limit_in_gb=0.0, pin_memory=False, min_offload_size=0):
        actual, _ = chunk_loss_module.chunk_loss_function(hidden, weight, labels, chunk_size=2, vocab_size=7)
        (actual * scale).backward()

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)


def test_chunk_loss_releases_cache_after_outer_hook_packs_saved_gradients(monkeypatch):
    """Opt-in cleanup runs only after ``ChunkLoss.apply`` finishes packing."""

    class _FakeParallelState:
        sp_enabled = False

    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: _FakeParallelState())
    events = []

    def pack(tensor):
        events.append("pack")
        return tensor

    def unpack(tensor):
        events.append("unpack")
        return tensor

    def release_cache(device):
        assert device.type == "cpu"
        assert events == ["pack", "pack"]
        events.append("release")

    monkeypatch.setattr(chunk_loss_module, "_release_accelerator_cache", release_cache)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.randint(0, 7, (1, 5))

    with saved_tensors_hooks(pack, unpack):
        loss, _ = chunk_loss_module.chunk_loss_function(
            hidden,
            weight,
            labels,
            chunk_size=2,
            vocab_size=7,
            release_cache=True,
        )
        assert events == ["pack", "pack", "release"]
        loss.backward()

    assert events == ["pack", "pack", "release", "unpack", "unpack"]


def test_chunk_loss_dispatch_forwards_chunk_size(monkeypatch):
    """The plain training path must honor the existing per-call chunk size."""
    captured = {}

    def fake_chunk_loss_function(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor(1.0), None

    monkeypatch.setattr(cross_entropy_module, "chunk_loss_function", fake_chunk_loss_function)
    loss, logits, aux = cross_entropy_module._chunk_loss_dispatch(
        hidden_states=torch.empty(1),
        weights=torch.empty(1),
        labels=torch.empty(1, dtype=torch.long),
        chunk_size=17,
    )

    assert loss.item() == 1.0
    assert logits is None and aux is None
    assert captured["chunk_size"] == 17


def test_chunk_loss_dispatch_preserves_positional_chunk_size(monkeypatch):
    """Do not inject a duplicate keyword for legacy positional callers."""
    captured = {}

    def fake_chunk_loss_function(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.tensor(1.0), None

    monkeypatch.setattr(cross_entropy_module, "chunk_loss_function", fake_chunk_loss_function)
    cross_entropy_module._chunk_loss_dispatch(torch.empty(1), torch.empty(1), torch.empty(1), 23)

    assert captured["args"][3] == 23
    assert "chunk_size" not in captured["kwargs"]


def test_release_accelerator_cache_synchronizes_before_emptying(monkeypatch):
    """The cleanup helper orders synchronization before allocator release."""
    events = []
    monkeypatch.setattr(chunk_loss_module, "stream_synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(chunk_loss_module, "empty_cache", lambda: events.append("empty_cache"))

    chunk_loss_module._release_accelerator_cache(SimpleNamespace(type="npu"))

    assert events == ["synchronize", "empty_cache"]


def test_release_accelerator_cache_is_noop_on_cpu(monkeypatch):
    """CPU tests and inference must not touch an accelerator namespace."""
    events = []
    monkeypatch.setattr(chunk_loss_module, "stream_synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(chunk_loss_module, "empty_cache", lambda: events.append("empty_cache"))

    chunk_loss_module._release_accelerator_cache(torch.device("cpu"))

    assert events == []


def test_chunk_loss_factory_binds_release_cache_config():
    """OpSlot dispatch carries the public ops-config cleanup opt-in."""
    assert "cross_entropy_loss_release_cache" in OpsImplementationConfig.__dataclass_fields__
    previous = get_ops_config()
    set_ops_config(SimpleNamespace(cross_entropy_loss_release_cache=True))
    try:
        kernel = cross_entropy_module._chunk_loss_causal_factory()
    finally:
        set_ops_config(previous)

    assert isinstance(kernel, partial)
    assert kernel.func is cross_entropy_module._chunk_loss_dispatch
    assert kernel.keywords["release_cache"] is True
