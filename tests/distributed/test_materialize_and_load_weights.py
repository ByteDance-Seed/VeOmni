"""Unit tests for ``_materialize_and_load_weights``.

These tests pin down the dispatch contract that ``parallelize_model_fsdp2`` and
``parallelize_model_ddp`` (and therefore ``build_parallelize_model``) rely on:

* ``weights_path is None`` — ``to_empty`` + ``init_weights`` (random init).
* ``weights_path: str``    — single-snapshot load, routed to
                             ``rank0_load_and_broadcast_weights`` or
                             ``load_model_weights`` by ``broadcast_from_rank0``.
* ``should_skip_hf_weight_load`` — materialise only, for a checkpoint resume.

We mock the actual loader functions so the tests run on CPU without distributed
init or real checkpoints — what we're verifying here is *control flow*, not
weight content (covered separately by the FSDP2 equivalence tests).
"""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from veomni.distributed.torch_parallelize import _materialize_and_load_weights, parallelize_model_ddp
from veomni.utils.device import get_device_type


# ── Test fixtures: a model with named children and a fake init_weights ─────


class _Leaf(nn.Module):
    """Plain nn.Module leaf with an ``init_weights`` hook so the random-init
    branch can exercise it.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(4, 4)
        self.init_called = 0

    def init_weights(self) -> None:  # pragma: no cover - exercised via mock
        self.init_called += 1


class _Container(nn.Module):
    """A model with two named children directly attached."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Leaf("encoder")
        self.decoder = _Leaf("decoder")
        self.init_called = 0

    def init_weights(self) -> None:  # pragma: no cover - exercised via mock
        self.init_called += 1


# ── None branch: random init ───────────────────────────────────────────────


def test_none_weights_path_calls_to_empty_and_init(monkeypatch):
    """``weights_path=None`` → ``model.to_empty(device=...)`` then
    ``model.init_weights()`` — no loader functions invoked."""
    model = _Container()
    to_empty_calls: list[tuple[Any, Any]] = []
    init_calls: list[int] = []

    def _fake_to_empty(self, *, device):
        to_empty_calls.append((self, device))

    def _fake_init(self):
        init_calls.append(id(self))

    monkeypatch.setattr(nn.Module, "to_empty", _fake_to_empty, raising=False)
    monkeypatch.setattr(_Container, "init_weights", _fake_init, raising=False)

    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _materialize_and_load_weights(
        model=model,
        weights_path=None,
        materialize_device="cpu",
        should_skip_hf_weight_load=False,
        broadcast_from_rank0=False,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        fqn_to_index_mapping=None,
    )

    assert to_empty_calls == [(model, "cpu")]
    assert init_calls == [id(model)]
    load_model_weights_mock.assert_not_called()
    rank0_mock.assert_not_called()


# ── str branch: single-snapshot load ───────────────────────────────────────


def test_str_weights_path_with_rank0_broadcast(monkeypatch):
    """``weights_path='/snap'`` + ``broadcast_from_rank0=True`` →
    rank0 loader called once with the full ``model`` and snapshot path."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _materialize_and_load_weights(
        model=model,
        weights_path="/snap/full",
        materialize_device=get_device_type(),
        should_skip_hf_weight_load=False,
        broadcast_from_rank0=True,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=["embed.weight"],
        max_load_broadcast_size=15.0,
        fqn_to_index_mapping=None,
    )

    rank0_mock.assert_called_once()
    args, kwargs = rank0_mock.call_args
    # Positional args: model, weights_path, materialize_device.
    assert args[0] is model
    assert args[1] == "/snap/full"
    assert args[2] == get_device_type()
    assert kwargs["cpu_load_param_name"] == ["embed.weight"]
    assert kwargs["max_load_broadcast_size"] == 15.0
    assert kwargs["is_peft_model"] is False
    assert kwargs["adapter_path"] is None
    load_model_weights_mock.assert_not_called()


def test_str_weights_path_without_rank0_broadcast(monkeypatch):
    """``broadcast_from_rank0=False`` → falls back to per-rank
    ``load_model_weights`` (slow path)."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _materialize_and_load_weights(
        model=model,
        weights_path="/snap/full",
        materialize_device=get_device_type(),
        should_skip_hf_weight_load=False,
        broadcast_from_rank0=False,
        is_peft_model=True,
        adapter_path="/snap/lora",
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        fqn_to_index_mapping=None,
    )

    load_model_weights_mock.assert_called_once()
    args, kwargs = load_model_weights_mock.call_args
    assert args[0] is model
    assert args[1] == "/snap/full"
    assert kwargs["is_peft_model"] is True
    assert kwargs["adapter_path"] == "/snap/lora"
    rank0_mock.assert_not_called()


def test_forwards_fqn_to_index_mapping(monkeypatch):
    """The caller's shard index reaches the loader untouched."""
    load_model_weights_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)

    mapping: Mapping[str, int] = {"linear.weight": 0}
    _materialize_and_load_weights(
        model=_Container(),
        weights_path="/snap/full",
        materialize_device="cpu",
        should_skip_hf_weight_load=False,
        broadcast_from_rank0=False,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        fqn_to_index_mapping=mapping,
    )

    assert load_model_weights_mock.call_args.kwargs["fqn_to_index_mapping"] is mapping


# ── DDP path: should_skip_hf_weight_load must reach the load step ──────────


def _frozen_meta_leaf() -> _Leaf:
    """A fully-frozen meta-init leaf: ``parallelize_model_ddp`` materialises it
    and then returns before the DDP wrap, so no process group is needed."""
    with torch.device("meta"):
        model = _Leaf("frozen")
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize("should_skip", [True, False])
def test_ddp_honors_should_skip_hf_weight_load(monkeypatch, should_skip):
    """A distributed-checkpoint resume must not re-read the HF snapshot under
    DDP, but the params still have to leave the meta device."""
    # A real loader materialises as it fills, and ``parallelize_model_ddp``
    # re-checks for meta params afterwards, so the stub has to materialise too.
    load_model_weights_mock = MagicMock(side_effect=lambda model, *a, **k: model.to_empty(device="cpu"))
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    model = parallelize_model_ddp(
        model=_frozen_meta_leaf(),
        weights_path="/snap/full",
        should_skip_hf_weight_load=should_skip,
        init_device="meta",
    )

    assert load_model_weights_mock.call_count == (0 if should_skip else 1)
    # Nothing else materialises these params, so a meta one here would blow up on
    # the first forward -- on the resume path because the loader is skipped, and
    # on the load path if the loader ever left one behind.
    assert not any(param.is_meta for param in model.parameters())


# ── DDP path: buffers are never broadcast (FSDP2 / HSDP parity) ────────────


class _BufferLeaf(nn.Module):
    """Both buffer flavours, including BatchNorm's mutable running stats."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(4)  # persistent running stats
        self.inner = nn.Module()
        self.inner.register_buffer("inv_freq", torch.ones(2), persistent=False)
        self.register_buffer("position_ids", torch.arange(3), persistent=False)

    def init_weights(self) -> None:  # pragma: no cover - not exercised here
        pass


def test_ddp_never_broadcasts_buffers(monkeypatch):
    """DDP must not sync buffers, so a module behaves the same whichever
    ``dp_mode`` the config picks: ``fully_shard`` syncs none either, and DDP's
    in-place pre-forward ``copy_`` would corrupt a buffer saved for backward by
    a module owning more than one graph node. Even BatchNorm's running stats stay
    out — the rank0 broadcast discards the other ranks' statistics rather than
    aggregating them, so ``SyncBatchNorm`` is the fix, not this flag.
    """
    captured: dict[str, Any] = {}

    def _fake_ddp(module, **kwargs):
        captured["kwargs"] = kwargs
        return module

    monkeypatch.setattr("veomni.distributed.torch_parallelize.DDP", _fake_ddp)

    parallelize_model_ddp(model=_BufferLeaf(), weights_path=None)

    assert captured["kwargs"]["broadcast_buffers"] is False
