"""CPU 2-rank contract for the grouped nested FSDP gated-norm leaf."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import (
    GROUPED_NESTED_FSDP_LEAF_ATTR,
    _iter_grouped_nested_fsdp_leaves,
    build_parallelize_model,
)


HIDDEN = 128


class _MarkedGatedNorm(nn.Module):
    _veomni_kcp_requires_grouped_nested_fsdp_leaf = True

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def enable_kcp_fsdp_contract(self, implementation: str) -> None:
        if implementation == "kcp" and getattr(self, "_veomni_kcp_requires_grouped_nested_fsdp_leaf", False):
            self._veomni_grouped_nested_fsdp_leaf = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


class _PlainGatedNorm(nn.Module):
    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


class _ToyDecoderLayer(nn.Module):
    def __init__(self, hidden: int = HIDDEN, marked: bool = True, implementation: str = "disabled") -> None:
        super().__init__()
        norm_cls = _MarkedGatedNorm if marked else _PlainGatedNorm
        self.norm = norm_cls(hidden)
        if marked:
            self.norm.enable_kcp_fsdp_contract(implementation)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden_states))


class _ToyModel(nn.Module):
    _no_split_modules = ["_ToyDecoderLayer"]

    def __init__(
        self,
        n_layers: int = 2,
        hidden: int = HIDDEN,
        marked: bool = True,
        implementation: str = "disabled",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_ToyDecoderLayer(hidden, marked=marked, implementation=implementation) for _ in range(n_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states.sum()

    def init_weights(self) -> None:
        for parameter in self.parameters():
            parameter.data.fill_(0.25)


class _ToyMixedPrecisionIgnoredModel(_ToyModel):
    def get_ignore_modules_in_mixed_precision(self) -> tuple[type[nn.Module], ...]:
        return (_MarkedGatedNorm,)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.full_tensor().detach().cpu()
    return value.detach().cpu()


def _wrap(model: _ToyModel, mixed_precision: MixedPrecisionConfig | None = None) -> nn.Module:
    if mixed_precision is None:
        mixed_precision = MixedPrecisionConfig(enable=False)
    return build_parallelize_model(
        model,
        init_device="meta",
        weights_path=None,
        mixed_precision=mixed_precision,
        enable_gradient_checkpointing=False,
        basic_modules=[],
        enable_forward_prefetch=False,
        broadcast_model_weights_from_rank0=False,
    )


def test_collector_finds_marked_modules_only() -> None:
    marked = _ToyModel(marked=True, implementation="kcp")
    plain = _ToyModel(marked=False)
    leaves = _iter_grouped_nested_fsdp_leaves(marked)
    assert len(leaves) == 2
    assert all(getattr(module, GROUPED_NESTED_FSDP_LEAF_ATTR) for module in leaves)
    assert _iter_grouped_nested_fsdp_leaves(plain) == []


def test_collector_preserves_cp_off_and_state_passing_layout() -> None:
    assert _iter_grouped_nested_fsdp_leaves(_ToyModel(marked=True, implementation="disabled")) == []
    assert _iter_grouped_nested_fsdp_leaves(_ToyModel(marked=True, implementation="state_passing_lossless")) == []


def _run_grouped_matches_parent_fsdp(
    rank: int,
    world_size: int,
    port: int,
    ignore_marked_in_mixed_precision: bool,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        init_parallel_state(
            dp_size=world_size,
            dp_shard_size=world_size,
            dp_mode="fsdp2",
            device_type="cpu",
            name="grouped_nested_fsdp_leaf",
        )
        torch.manual_seed(17)
        if ignore_marked_in_mixed_precision:
            model_cls = _ToyMixedPrecisionIgnoredModel
            mixed_precision = MixedPrecisionConfig(
                enable=True,
                param_dtype=None,
                reduce_dtype=None,
                output_dtype=None,
                cast_forward_inputs=False,
            )
            reference_model = model_cls(marked=True, implementation="disabled")
        else:
            model_cls = _ToyModel
            mixed_precision = MixedPrecisionConfig(enable=False)
            reference_model = model_cls(marked=False)
        reference = _wrap(reference_model, mixed_precision)
        candidate = _wrap(model_cls(marked=True, implementation="kcp"), mixed_precision)

        marked = [module for module in candidate.modules() if getattr(module, GROUPED_NESTED_FSDP_LEAF_ATTR, False)]
        assert len(marked) == 2
        assert all(isinstance(module, FSDPModule) for module in marked)
        assert all(isinstance(layer, FSDPModule) for layer in candidate.layers)
        assert all(layer.norm in layer._fsdp_modules for layer in candidate.layers)

        seen_types: list[str] = []

        def _record_weight(module: nn.Module, _inputs: tuple) -> None:
            weight = module.weight
            seen_types.append(type(weight).__name__)
            assert not isinstance(weight, DTensor)
            assert type(weight).__name__ in {"Tensor", "Parameter"}
            assert tuple(weight.shape) == (HIDDEN,)

        for module in marked:
            module.register_forward_pre_hook(_record_weight)

        inputs = torch.randn(2, 8, HIDDEN)
        ref_out = reference(inputs.clone())
        cand_out = candidate(inputs.clone())
        torch.testing.assert_close(ref_out, cand_out, rtol=0, atol=0)
        assert len(seen_types) == 2

        ref_out.backward()
        cand_out.backward()
        for ref_layer, cand_layer in zip(reference.layers, candidate.layers):
            torch.testing.assert_close(
                _full_tensor(ref_layer.norm.weight.grad),
                _full_tensor(cand_layer.norm.weight.grad),
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                _full_tensor(ref_layer.proj.weight.grad),
                _full_tensor(cand_layer.proj.weight.grad),
                rtol=1e-6,
                atol=1e-6,
            )

        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
        cand_opt = torch.optim.SGD(candidate.parameters(), lr=0.1)
        ref_opt.step()
        cand_opt.step()
        for ref_layer, cand_layer in zip(reference.layers, candidate.layers):
            torch.testing.assert_close(
                _full_tensor(ref_layer.norm.weight),
                _full_tensor(cand_layer.norm.weight),
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        clear_parallel_state()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_grouped_nested_leaf_matches_parent_fsdp_cpu2() -> None:
    world_size = 2
    mp.spawn(
        _run_grouped_matches_parent_fsdp,
        args=(world_size, _free_port(), False),
        nprocs=world_size,
        join=True,
    )


def test_grouped_nested_leaf_respects_mixed_precision_ignore_cpu2() -> None:
    world_size = 2
    mp.spawn(
        _run_grouped_matches_parent_fsdp,
        args=(world_size, _free_port(), True),
        nprocs=world_size,
        join=True,
    )
