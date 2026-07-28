"""Correctness tests for two-dimensional GDN sequence parallelism."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from veomni.distributed.context_parallel.gdn_sp import (
    cp_gather_sequence,
    cp_scatter_sequence,
    derive_gdn_local_cu_seqlens,
    resolve_gdn_parallel_plan,
    scale_cp_repeated_parameter_grad,
)
from veomni.distributed.context_parallel.packed_sharding import build_packed_cp_partition
from veomni.distributed.context_parallel.sharding import balanced_cp_restore, balanced_cp_slice


@dataclass
class _FakeParallelState:
    ulysses_size: int
    cp_size: int
    ulysses_group: object
    cp_group: object
    ulysses_rank: int = 0
    cp_rank: int = 0

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1


def test_gdn_plan_keeps_ulysses_heads_and_cp_sequence_separate():
    state = _FakeParallelState(
        ulysses_size=8,
        cp_size=4,
        ulysses_group=object(),
        cp_group=object(),
    )
    with patch("veomni.distributed.context_parallel.gdn_sp.get_parallel_state", return_value=state):
        plan = resolve_gdn_parallel_plan()

    assert plan.ulysses_size == 8
    assert plan.cp_size == 4
    assert plan.head_parallel_enabled
    assert plan.cp_sequence_enabled
    assert plan.cp_seq_enabled == plan.cp_sequence_enabled


def _toy_forward(
    local_input: torch.Tensor,
    *,
    pre_weight: torch.Tensor,
    core_weight: torch.Tensor,
    z_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    out_weight: torch.Tensor,
    cp_group,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    projected = local_input @ pre_weight
    z = torch.sigmoid(local_input @ z_weight)

    full_projected = cp_gather_sequence(
        projected,
        cp_group=cp_group,
        cp_size=cp_size,
        seq_dim=1,
    )
    repeated_core_weight = scale_cp_repeated_parameter_grad(core_weight, cp_size=cp_size)
    full_core = torch.cumsum(full_projected, dim=1) @ repeated_core_weight
    local_core = cp_scatter_sequence(
        full_core,
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_dim=1,
    )

    # These operations consume only this rank's output shard. Their parameters
    # are not part of the repeated full-sequence core and must not be scaled.
    normalized = local_core * norm_weight
    return (normalized * z) @ out_weight


def _dense_toy_forward(
    full_input: torch.Tensor,
    *,
    pre_weight: torch.Tensor,
    core_weight: torch.Tensor,
    z_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    out_weight: torch.Tensor,
) -> torch.Tensor:
    projected = full_input @ pre_weight
    core = torch.cumsum(projected, dim=1) @ core_weight
    z = torch.sigmoid(full_input @ z_weight)
    return (core * norm_weight * z) @ out_weight


def _gdn_dense_parity_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue) -> None:
    try:
        store = dist.FileStore(file_name, world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
        group = dist.distributed_c10d._get_default_group()

        torch.manual_seed(1234)
        full_input = torch.randn(1, 8, 3, dtype=torch.float64)
        target = torch.randn(1, 8, 2, dtype=torch.float64)
        initial_parameters = {
            "pre_weight": torch.randn(3, 4, dtype=torch.float64),
            "core_weight": torch.randn(4, 4, dtype=torch.float64),
            "z_weight": torch.randn(3, 4, dtype=torch.float64),
            "norm_weight": torch.randn(4, dtype=torch.float64),
            "out_weight": torch.randn(4, 2, dtype=torch.float64),
        }

        reference_input = full_input.detach().clone().requires_grad_(True)
        reference_parameters = {
            name: value.detach().clone().requires_grad_(True) for name, value in initial_parameters.items()
        }
        reference_output = _dense_toy_forward(reference_input, **reference_parameters)
        (reference_output * target).sum().backward()

        local_input = (
            balanced_cp_slice(full_input, cp_size=world_size, cp_rank=rank, dim=1).detach().requires_grad_(True)
        )
        local_target = balanced_cp_slice(target, cp_size=world_size, cp_rank=rank, dim=1)
        local_parameters = {
            name: value.detach().clone().requires_grad_(True) for name, value in initial_parameters.items()
        }
        local_output = _toy_forward(
            local_input,
            **local_parameters,
            cp_group=group,
            cp_size=world_size,
            cp_rank=rank,
        )
        (local_output * local_target).sum().backward()

        expected_local_output = balanced_cp_slice(
            reference_output.detach(),
            cp_size=world_size,
            cp_rank=rank,
            dim=1,
        )
        torch.testing.assert_close(local_output, expected_local_output, atol=1e-10, rtol=1e-10)
        expected_local_input_grad = balanced_cp_slice(
            reference_input.grad,
            cp_size=world_size,
            cp_rank=rank,
            dim=1,
        )
        torch.testing.assert_close(local_input.grad, expected_local_input_grad, atol=1e-10, rtol=1e-10)

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output.detach(), group=group)
        restored_output = balanced_cp_restore(torch.cat(gathered_outputs, dim=1), cp_size=world_size, dim=1)
        torch.testing.assert_close(restored_output, reference_output.detach(), atol=1e-10, rtol=1e-10)

        for name, parameter in local_parameters.items():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameters[name].grad,
                atol=1e-10,
                rtol=1e-10,
                msg=lambda message, parameter_name=name: (
                    f"{message}\ncollective parameter-gradient mismatch for {parameter_name}"
                ),
            )

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        if dist.is_initialized():
            dist.destroy_process_group()
        errors.put(repr(exc))


def test_cp_repeated_gdn_matches_dense_output_input_and_parameter_grads():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "gdn-cp-pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(
                target=_gdn_dense_parity_worker,
                args=(rank, world_size, file_name, errors),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            error = errors.get(timeout=5)
            assert error is None, error


def _packed_cumsum(tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    points = cu_seqlens.tolist()
    return torch.cat(
        [tensor.narrow(1, int(start), int(end - start)).cumsum(dim=1) for start, end in zip(points[:-1], points[1:])],
        dim=1,
    )


def _gloo_all_to_all_tensor(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group,
    async_op: bool = False,
) -> torch.Tensor:
    """Reference all-to-all for CPU tests because Gloo lacks ``all_to_all``."""
    if async_op:
        raise NotImplementedError("The test-only Gloo all-to-all does not support async collectives.")
    world_size = dist.get_world_size(group)
    destination_rank = dist.get_rank(group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    destination_chunks = [
        torch.tensor_split(source, world_size, dim=scatter_dim)[destination_rank].contiguous() for source in gathered
    ]
    return torch.cat(destination_chunks, dim=gather_dim).contiguous()


def _packed_cp_parity_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue) -> None:
    try:
        store = dist.FileStore(file_name, world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
        group = dist.distributed_c10d._get_default_group()

        torch.manual_seed(4321)
        global_cu = torch.tensor([0, 8, 20], dtype=torch.int32)
        full_input = torch.randn(1, 20, 3, dtype=torch.float64)
        target = torch.randn(1, 20, 2, dtype=torch.float64)
        initial_weight = torch.randn(3, 2, dtype=torch.float64)

        reference_input = full_input.detach().clone().requires_grad_(True)
        reference_weight = initial_weight.detach().clone().requires_grad_(True)
        reference_output = _packed_cumsum(reference_input, global_cu) @ reference_weight
        (reference_output * target).sum().backward()

        partition = build_packed_cp_partition(
            global_cu,
            cp_size=world_size,
            cp_rank=rank,
        )
        local_input = full_input.index_select(1, partition.token_indices).detach().requires_grad_(True)
        local_target = target.index_select(1, partition.token_indices)
        local_weight = initial_weight.detach().clone().requires_grad_(True)
        _, cp_local_cu = derive_gdn_local_cu_seqlens(
            global_cu,
            ulysses_size=1,
            cp_size=world_size,
        )

        full_gathered = cp_gather_sequence(
            local_input,
            cp_group=group,
            cp_size=world_size,
            seq_dim=1,
            cp_local_cu_seqlens=cp_local_cu,
        )
        repeated_weight = scale_cp_repeated_parameter_grad(local_weight, cp_size=world_size)
        full_output = _packed_cumsum(full_gathered, global_cu) @ repeated_weight
        local_output = cp_scatter_sequence(
            full_output,
            cp_group=group,
            cp_size=world_size,
            cp_rank=rank,
            seq_dim=1,
            cp_local_cu_seqlens=cp_local_cu,
        )
        (local_output * local_target).sum().backward()

        torch.testing.assert_close(
            local_input.grad,
            reference_input.grad.index_select(1, partition.token_indices),
            atol=1e-10,
            rtol=1e-10,
        )

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output.detach(), group=group)
        restored_output = torch.empty_like(reference_output)
        for cp_rank, output_shard in enumerate(gathered_outputs):
            cp_partition = build_packed_cp_partition(
                global_cu,
                cp_size=world_size,
                cp_rank=cp_rank,
            )
            restored_output.index_copy_(1, cp_partition.token_indices, output_shard)
        torch.testing.assert_close(restored_output, reference_output.detach(), atol=1e-10, rtol=1e-10)

        dist.all_reduce(local_weight.grad, op=dist.ReduceOp.SUM, group=group)
        torch.testing.assert_close(local_weight.grad, reference_weight.grad, atol=1e-10, rtol=1e-10)

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        if dist.is_initialized():
            dist.destroy_process_group()
        errors.put(repr(exc))


def test_cp_packed_multisegment_output_input_and_parameter_grad_parity():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "gdn-packed-cp-pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(
                target=_packed_cp_parity_worker,
                args=(rank, world_size, file_name, errors),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            error = errors.get(timeout=5)
            assert error is None, error


def _fake_causal_conv1d(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
    cu_seqlens: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor]:
    kernel_size = weight.shape[-1]
    if cu_seqlens is None:
        segments = (x,)
    else:
        points = cu_seqlens.tolist()
        segments = tuple(x.narrow(1, int(start), int(end - start)) for start, end in zip(points[:-1], points[1:]))
    outputs = [
        F.conv1d(
            F.pad(segment.transpose(1, 2), (kernel_size - 1, 0)),
            weight[:, None, :],
            bias=bias,
            groups=x.shape[-1],
        ).transpose(1, 2)
        for segment in segments
    ]
    output = torch.cat(outputs, dim=1)
    if activation == "silu":
        output = F.silu(output)
    return (output,)


def _fake_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, None]:
    recurrent_input = value + 0.1 * query + 0.2 * key + g.unsqueeze(-1) + beta.unsqueeze(-1)
    if cu_seqlens is None:
        return torch.cumsum(recurrent_input, dim=1), None
    return _packed_cumsum(recurrent_input, cu_seqlens), None


def _model_integration_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue) -> None:
    try:
        from types import SimpleNamespace

        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeGatedDeltaNet,
        )

        store = dist.FileStore(file_name, world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
        group = dist.distributed_c10d._get_default_group()

        torch.manual_seed(2026)
        config = SimpleNamespace(
            hidden_size=4,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            dtype=torch.float64,
        )
        layer = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).double()
        layer.causal_conv1d_fn = _fake_causal_conv1d
        layer.chunk_gated_delta_rule = _fake_gated_delta_rule

        full_input = torch.randn(1, 8, config.hidden_size, dtype=torch.float64)
        target = torch.randn_like(full_input)
        global_cu_seqlens = torch.tensor([0, full_input.shape[1]], dtype=torch.int32)

        reference_input = full_input.detach().clone().requires_grad_(True)
        no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
        with patch("veomni.distributed.context_parallel.gdn_sp.get_parallel_state", return_value=no_sp_state):
            reference_output = layer(
                reference_input,
                attention_mask=None,
                cu_seq_lens_q=global_cu_seqlens,
            )
            (reference_output * target).sum().backward()
        reference_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }
        layer.zero_grad(set_to_none=True)

        local_input = (
            balanced_cp_slice(full_input, cp_size=world_size, cp_rank=rank, dim=1).detach().requires_grad_(True)
        )
        local_target = balanced_cp_slice(target, cp_size=world_size, cp_rank=rank, dim=1)
        cp_state = SimpleNamespace(
            ulysses_enabled=False,
            cp_enabled=True,
            cp_group=group,
            cp_size=world_size,
            cp_rank=rank,
        )
        with patch("veomni.distributed.context_parallel.gdn_sp.get_parallel_state", return_value=cp_state):
            local_output = layer(
                local_input,
                attention_mask=None,
                cu_seq_lens_q=global_cu_seqlens,
            )
            (local_output * local_target).sum().backward()

        expected_local_input_grad = balanced_cp_slice(
            reference_input.grad,
            cp_size=world_size,
            cp_rank=rank,
            dim=1,
        )
        torch.testing.assert_close(local_input.grad, expected_local_input_grad, atol=1e-8, rtol=1e-7)

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output.detach(), group=group)
        restored_output = balanced_cp_restore(torch.cat(gathered_outputs, dim=1), cp_size=world_size, dim=1)
        torch.testing.assert_close(restored_output, reference_output.detach(), atol=1e-8, rtol=1e-7)

        for name, parameter in layer.named_parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter_grads[name],
                atol=1e-7,
                rtol=1e-5,
                msg=lambda message, parameter_name=name: (
                    f"{message}\nQwen3.5-MoE GDN parameter-gradient mismatch for {parameter_name}"
                ),
            )

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        if dist.is_initialized():
            dist.destroy_process_group()
        errors.put(repr(exc))


def test_qwen3_5_moe_gdn_cp_path_matches_dense_output_input_and_parameter_grads():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "qwen-gdn-cp-pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(
                target=_model_integration_worker,
                args=(rank, world_size, file_name, errors),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            error = errors.get(timeout=5)
            assert error is None, error


def _model_u2cp2_packed_worker(rank: int, world_size: int, file_name: str, errors: mp.Queue) -> None:
    try:
        from types import SimpleNamespace

        from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeGatedDeltaNet,
        )

        store = dist.FileStore(file_name, world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)

        ulysses_groups = (
            dist.new_group(ranks=[0, 1]),
            dist.new_group(ranks=[2, 3]),
        )
        cp_groups = (
            dist.new_group(ranks=[0, 2]),
            dist.new_group(ranks=[1, 3]),
        )
        ulysses_size = 2
        cp_size = 2
        cp_rank = rank // ulysses_size
        ulysses_rank = rank % ulysses_size
        ulysses_group = ulysses_groups[cp_rank]
        cp_group = cp_groups[ulysses_rank]

        torch.manual_seed(2027)
        config = SimpleNamespace(
            hidden_size=4,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=3,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            dtype=torch.float64,
        )
        layer = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).double()
        layer.causal_conv1d_fn = _fake_causal_conv1d
        layer.chunk_gated_delta_rule = _fake_gated_delta_rule

        global_cu_seqlens = torch.tensor([0, 16, 40], dtype=torch.int32)
        full_input = torch.randn(1, 40, config.hidden_size, dtype=torch.float64)
        target = torch.randn_like(full_input)

        reference_input = full_input.detach().clone().requires_grad_(True)
        no_sp_state = SimpleNamespace(ulysses_enabled=False, cp_enabled=False)
        with patch("veomni.distributed.context_parallel.gdn_sp.get_parallel_state", return_value=no_sp_state):
            reference_output = layer(
                reference_input,
                attention_mask=None,
                cu_seq_lens_q=global_cu_seqlens,
            )
            (reference_output * target).sum().backward()
        reference_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }
        layer.zero_grad(set_to_none=True)

        partition = build_packed_cp_partition(
            global_cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
            ulysses_size=ulysses_size,
            ulysses_rank=ulysses_rank,
        )
        local_input = full_input.index_select(1, partition.token_indices).detach().requires_grad_(True)
        local_target = target.index_select(1, partition.token_indices)
        hybrid_state = SimpleNamespace(
            ulysses_enabled=True,
            cp_enabled=True,
            ulysses_group=ulysses_group,
            ulysses_size=ulysses_size,
            ulysses_rank=ulysses_rank,
            cp_group=cp_group,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        with (
            patch("veomni.distributed.context_parallel.gdn_sp.get_parallel_state", return_value=hybrid_state),
            patch(
                "veomni.distributed.sequence_parallel.ulysses.all_to_all_tensor",
                new=_gloo_all_to_all_tensor,
            ),
        ):
            local_output = layer(
                local_input,
                attention_mask=None,
                cu_seq_lens_q=global_cu_seqlens,
            )
            (local_output * local_target).sum().backward()

        torch.testing.assert_close(
            local_input.grad,
            reference_input.grad.index_select(1, partition.token_indices),
            atol=1e-8,
            rtol=1e-7,
        )

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output.detach())
        restored_output = torch.empty_like(reference_output)
        for peer_rank, output_shard in enumerate(gathered_outputs):
            peer_cp_rank = peer_rank // ulysses_size
            peer_ulysses_rank = peer_rank % ulysses_size
            peer_partition = build_packed_cp_partition(
                global_cu_seqlens,
                cp_size=cp_size,
                cp_rank=peer_cp_rank,
                ulysses_size=ulysses_size,
                ulysses_rank=peer_ulysses_rank,
            )
            restored_output.index_copy_(1, peer_partition.token_indices, output_shard)
        torch.testing.assert_close(restored_output, reference_output.detach(), atol=1e-8, rtol=1e-7)

        for name, parameter in layer.named_parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter_grads[name],
                atol=1e-7,
                rtol=1e-5,
                msg=lambda message, parameter_name=name: (
                    f"{message}\nU2CP2 packed Qwen3.5-MoE GDN parameter-gradient mismatch for {parameter_name}"
                ),
            )

        dist.destroy_process_group()
        errors.put(None)
    except Exception as exc:  # noqa: BLE001
        if dist.is_initialized():
            dist.destroy_process_group()
        errors.put(repr(exc))


def test_qwen3_5_moe_gdn_u2cp2_packed_matches_dense_output_input_and_parameter_grads():
    world_size = 4
    with tempfile.TemporaryDirectory() as tmp:
        file_name = os.path.join(tmp, "qwen-gdn-u2cp2-packed-pg")
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        processes = [
            ctx.Process(
                target=_model_u2cp2_packed_worker,
                args=(rank, world_size, file_name, errors),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=180)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        for _ in range(world_size):
            error = errors.get(timeout=5)
            assert error is None, error
