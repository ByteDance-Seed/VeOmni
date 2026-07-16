"""Unit tests for ``_apply_weights_load_step`` weights_path dispatch (D2.2).

These tests pin down the three-branch dispatch contract that
``parallelize_model_fsdp2`` (and therefore ``build_parallelize_model``)
relies on:

* ``weights_path is None``  — ``to_empty`` + ``init_weights`` (random init).
* ``weights_path: str``     — single-snapshot load via ``rank0_*`` /
                              ``load_model_weights`` (the path all
                              existing single-model trainers hit).
* ``weights_path: Mapping`` — per-named-child snapshot load with strict
                              bijection over ``model.named_children()``.
                              Missing or extra keys ``KeyError`` up front.
                              Incompatible with ``is_peft_model=True``.

We mock the actual loader functions so the tests run on CPU without
distributed init or real checkpoints — what we're verifying here is
*control flow*, not weight content (covered separately by the existing
FSDP2 equivalence tests + D2.3's smoke tests under real FSDP).
"""

from __future__ import annotations

from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from veomni.distributed.torch_parallelize import (
    _apply_weights_load_step,
    _load_one,
    _resolve_weights_path_mapping,
)
from veomni.utils.device import get_device_type


# ── Test fixtures: a model with named children and a fake init_weights ─────


class _Leaf(nn.Module):
    """Plain nn.Module leaf with an ``init_weights`` hook so the
    ``None`` / partial-mapping branches can exercise it.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(4, 4)
        self.init_called = 0

    def init_weights(self) -> None:  # pragma: no cover - exercised via mock
        self.init_called += 1


class _Container(nn.Module):
    """Fake ``OmniModel`` shape: two named children directly attached."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Leaf("encoder")
        self.decoder = _Leaf("decoder")
        self.init_called = 0

    def init_weights(self) -> None:  # pragma: no cover - exercised via mock
        self.init_called += 1


# ── _resolve_weights_path_mapping ─────────────────────────────────────────


def test_resolve_mapping_full_coverage_ordered():
    """Strict bijection: every named_child has an entry → returned in
    mapping declaration order so the load log is predictable."""
    model = _Container()
    mapping = {"decoder": "/p/dec", "encoder": "/p/enc"}  # reversed vs. children order

    loaded = _resolve_weights_path_mapping(model, mapping)

    assert [name for name, _, _ in loaded] == ["decoder", "encoder"]
    assert [path for _, _, path in loaded] == ["/p/dec", "/p/enc"]
    assert all(child is getattr(model, name) for name, child, _ in loaded)


def test_resolve_mapping_unknown_key_raises_keyerror():
    """Unknown keys raise loud KeyError that names every offender + the
    available children — so YAML typos surface immediately."""
    model = _Container()
    mapping = {"encoder": "/p/enc", "decoder": "/p/dec", "typo_module": "/p/x"}

    with pytest.raises(KeyError) as excinfo:
        _resolve_weights_path_mapping(model, mapping)

    msg = str(excinfo.value)
    assert "typo_module" in msg
    assert "unknown" in msg.lower()
    assert "encoder" in msg  # available children listed too
    assert "decoder" in msg


def test_resolve_mapping_missing_children_raises_keyerror():
    """Direct children NOT covered by the mapping raise — the strict
    bijection prevents D2.3 from accidentally leaving a sub-module
    unloaded after FSDP wrap (which would surface much later as a NaN
    forward or random behaviour)."""
    model = _Container()
    mapping = {"encoder": "/p/enc"}  # decoder missing

    with pytest.raises(KeyError) as excinfo:
        _resolve_weights_path_mapping(model, mapping)

    msg = str(excinfo.value)
    assert "decoder" in msg
    assert "missing" in msg.lower()


def test_resolve_mapping_unknown_and_missing_both_reported():
    """When BOTH sides of the bijection break, the error names both
    offenders so the user fixes them in one round-trip."""
    model = _Container()
    mapping = {"encoder": "/p/enc", "typo": "/p/x"}  # decoder missing AND typo unknown

    with pytest.raises(KeyError) as excinfo:
        _resolve_weights_path_mapping(model, mapping)

    msg = str(excinfo.value)
    assert "decoder" in msg  # missing
    assert "typo" in msg  # unknown


# ── _apply_weights_load_step: None branch ─────────────────────────────────


def test_apply_load_step_none_calls_to_empty_and_init(monkeypatch):
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

    _apply_weights_load_step(
        model=model,
        weights_path=None,
        materialize_device="cpu",
        broadcast_from_rank0=False,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        distribute_tensor_fn=lambda *a, **k: None,
    )

    assert to_empty_calls == [(model, "cpu")]
    assert init_calls == [id(model)]
    load_model_weights_mock.assert_not_called()
    rank0_mock.assert_not_called()


# ── _apply_weights_load_step: str branch (legacy single-model path) ───────


def test_apply_load_step_str_with_rank0_broadcast(monkeypatch):
    """``weights_path='/snap'`` + ``broadcast_from_rank0=True`` →
    rank0 loader called once with the full ``model`` and snapshot path."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _apply_weights_load_step(
        model=model,
        weights_path="/snap/full",
        materialize_device=get_device_type(),
        broadcast_from_rank0=True,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=["embed.weight"],
        max_load_broadcast_size=15.0,
        distribute_tensor_fn=lambda *a, **k: None,
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


def test_apply_load_step_str_without_rank0_broadcast(monkeypatch):
    """``broadcast_from_rank0=False`` → falls back to per-rank
    ``load_model_weights`` (slow path)."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _apply_weights_load_step(
        model=model,
        weights_path="/snap/full",
        materialize_device=get_device_type(),
        broadcast_from_rank0=False,
        is_peft_model=True,
        adapter_path="/snap/lora",
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        distribute_tensor_fn=lambda *a, **k: None,
    )

    load_model_weights_mock.assert_called_once()
    args, kwargs = load_model_weights_mock.call_args
    assert args[0] is model
    assert args[1] == "/snap/full"
    assert kwargs["is_peft_model"] is True
    assert kwargs["adapter_path"] == "/snap/lora"
    rank0_mock.assert_not_called()


# ── _apply_weights_load_step: Mapping branch (V2 multi-snapshot) ──────────


def test_apply_load_step_mapping_calls_loader_per_child(monkeypatch):
    """Full-coverage Mapping → loader called once per (sub_module, path),
    in the mapping's declared order; no random-init fallback ever fires."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)
    # Spy on to_empty / init_weights — must NOT fire under strict bijection.
    to_empty_mock = MagicMock()
    init_weights_mock = MagicMock()
    monkeypatch.setattr(nn.Module, "to_empty", to_empty_mock, raising=False)
    monkeypatch.setattr(_Container, "init_weights", init_weights_mock, raising=False)
    monkeypatch.setattr(_Leaf, "init_weights", init_weights_mock, raising=False)

    mapping: Mapping[str, str] = {"encoder": "/p/enc", "decoder": "/p/dec"}
    _apply_weights_load_step(
        model=model,
        weights_path=mapping,
        materialize_device="cpu",
        broadcast_from_rank0=True,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        distribute_tensor_fn=lambda *a, **k: None,
    )

    # rank0_load called once per sub-module, in mapping iteration order.
    assert rank0_mock.call_count == 2
    targets_in_order = [(call.args[0], call.args[1]) for call in rank0_mock.call_args_list]
    assert targets_in_order == [
        (model.encoder, "/p/enc"),
        (model.decoder, "/p/dec"),
    ]
    load_model_weights_mock.assert_not_called()
    # Strict bijection: no random-init paths should fire.
    init_weights_mock.assert_not_called()
    to_empty_mock.assert_not_called()


def test_apply_load_step_mapping_unknown_key_raises(monkeypatch):
    """Unknown sub-module name in mapping → KeyError before any loader runs."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    with pytest.raises(KeyError, match="bogus"):
        _apply_weights_load_step(
            model=model,
            weights_path={"encoder": "/p/enc", "decoder": "/p/dec", "bogus": "/p/x"},
            materialize_device="cpu",
            broadcast_from_rank0=True,
            is_peft_model=False,
            adapter_path=None,
            cpu_load_param_name=None,
            max_load_broadcast_size=20.0,
            distribute_tensor_fn=lambda *a, **k: None,
        )

    load_model_weights_mock.assert_not_called()
    rank0_mock.assert_not_called()


def test_apply_load_step_mapping_missing_child_raises(monkeypatch):
    """Direct child not covered by the mapping → KeyError before any
    loader runs.  Prevents D2.3 from leaving a sub-module silently
    un-initialised after FSDP wrap."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    with pytest.raises(KeyError, match="decoder"):
        _apply_weights_load_step(
            model=model,
            weights_path={"encoder": "/p/enc"},  # decoder missing
            materialize_device="cpu",
            broadcast_from_rank0=True,
            is_peft_model=False,
            adapter_path=None,
            cpu_load_param_name=None,
            max_load_broadcast_size=20.0,
            distribute_tensor_fn=lambda *a, **k: None,
        )

    load_model_weights_mock.assert_not_called()
    rank0_mock.assert_not_called()


def test_apply_load_step_mapping_rejects_peft(monkeypatch):
    """``is_peft_model=True`` + Mapping → ``NotImplementedError`` before
    any loader runs.  PEFT semantics across heterogeneous OmniModel
    sub-modules aren't defined yet; refusing the combination prevents
    silent miswiring."""
    model = _Container()
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    with pytest.raises(NotImplementedError, match="is_peft_model=True"):
        _apply_weights_load_step(
            model=model,
            weights_path={"encoder": "/p/enc", "decoder": "/p/dec"},
            materialize_device="cpu",
            broadcast_from_rank0=True,
            is_peft_model=True,
            adapter_path="/some/lora",
            cpu_load_param_name=None,
            max_load_broadcast_size=20.0,
            distribute_tensor_fn=lambda *a, **k: None,
        )

    load_model_weights_mock.assert_not_called()
    rank0_mock.assert_not_called()


# ── _load_one selects the right loader ────────────────────────────────────


def test_load_one_routes_to_rank0_when_broadcast_enabled(monkeypatch):
    model = _Leaf("solo")
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _load_one(
        model=model,
        weights_path="/p/x",
        materialize_device="cpu",
        broadcast_from_rank0=True,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        distribute_tensor_fn=lambda *a, **k: None,
    )

    rank0_mock.assert_called_once()
    load_model_weights_mock.assert_not_called()


def test_load_one_routes_to_load_model_weights_otherwise(monkeypatch):
    model = _Leaf("solo")
    load_model_weights_mock = MagicMock()
    rank0_mock = MagicMock()
    monkeypatch.setattr("veomni.distributed.torch_parallelize.load_model_weights", load_model_weights_mock)
    monkeypatch.setattr("veomni.distributed.torch_parallelize.rank0_load_and_broadcast_weights", rank0_mock)

    _load_one(
        model=model,
        weights_path="/p/x",
        materialize_device="cpu",
        broadcast_from_rank0=False,
        is_peft_model=False,
        adapter_path=None,
        cpu_load_param_name=None,
        max_load_broadcast_size=20.0,
        distribute_tensor_fn=lambda *a, **k: None,
    )

    load_model_weights_mock.assert_called_once()
    rank0_mock.assert_not_called()
