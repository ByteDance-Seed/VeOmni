import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import veomni.distributed.context_parallel.gdn_kcp as gdn_kcp_module
import veomni.distributed.context_parallel.gdn_lossless as gdn_lossless_module
from veomni.distributed.context_parallel.gdn_kcp import (
    _deterministic_reduce_scatter_sum,
    all_gather_affine_hm,
    local_affine_summary_fused_torch,
    resolve_kcp_initial_state,
    unpack_affine_hm,
)
from veomni.distributed.context_parallel.gdn_lossless import (
    attach_state_dependency,
    compile_gdn_lossless_runtime_plan,
    make_state_participation,
    owned_to_physical,
    physical_to_owned,
)
from veomni.distributed.context_parallel.gdn_runtime import make_gdn_cp_runtime_observer
from veomni.ops.kernels.gated_delta_rule.normalization import producer_dtype_l2norm


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _global_inputs(tokens: int):
    torch.manual_seed(314)
    key = torch.randn(1, tokens, 1, 2, dtype=torch.bfloat16) * 0.05
    value = torch.randn(1, tokens, 1, 2, dtype=torch.float32) * 0.05
    g = -torch.rand(1, tokens, 1, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(1, tokens, 1, dtype=torch.float32))
    return key, value, g, beta


def _run_kcp_full_grad(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        local_tokens = 64
        global_tensors = _global_inputs(local_tokens * world_size)
        start = rank * local_tokens
        end = start + local_tokens
        local_raw = tuple(tensor[:, start:end].clone().detach().requires_grad_() for tensor in global_tensors)
        local = (producer_dtype_l2norm(local_raw[0]), *local_raw[1:])
        plan = compile_gdn_lossless_runtime_plan(
            [local_tokens * world_size],
            cp_group=dist.group.WORLD,
            ulysses_size=1,
        )
        cu = torch.tensor([0, local_tokens], dtype=torch.int32)
        initial_state = resolve_kcp_initial_state(
            *local,
            plan=plan,
            cp_group=dist.group.WORLD,
            cu_seqlens=cu,
            use_qk_l2norm=False,
            affine_impl="torch_reference",
        )
        local_hm = local_affine_summary_fused_torch(*local, cu_seqlens=cu, use_qk_l2norm=False)
        he, matrix = unpack_affine_hm(local_hm, v_dim=2)
        final_state = torch.einsum("nhki,nhiv->nhkv", matrix, initial_state) + he
        terminal_rank = plan.local.successor_rank is None
        loss = final_state.sum() if terminal_rank else final_state.sum() * 0
        loss.backward()

        oracle_raw = tuple(tensor.clone().detach().requires_grad_() for tensor in global_tensors)
        oracle = (producer_dtype_l2norm(oracle_raw[0]), *oracle_raw[1:])
        oracle_hm = local_affine_summary_fused_torch(*oracle, use_qk_l2norm=False)
        oracle_final, _ = unpack_affine_hm(oracle_hm, v_dim=2)
        oracle_final.sum().backward()
        for actual, expected in zip(local_raw, oracle_raw):
            torch.testing.assert_close(actual.grad, expected.grad[:, start:end], rtol=2e-4, atol=2e-5)
        if terminal_rank:
            torch.testing.assert_close(final_state, oracle_final, rtol=2e-4, atol=2e-5)

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
def test_kcp_gloo_cp_ladder_matches_monolithic_full_grad(world_size: int):
    mp.spawn(_run_kcp_full_grad, args=(world_size, _free_port()), nprocs=world_size, join=True)


def _run_deterministic_reduce_scatter(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # Channel 0 makes every source/destination route observable. Channel 1
        # is intentionally non-associative in fp32, so the expected value is
        # constructed with the same explicit source-rank left fold as the
        # production helper. Together they cover routing and repeatable
        # floating-point accumulation without relying on an exact integer sum.
        source_contributions = (1.0e20, 1.0, -1.0e20, 3.0)
        grad_ag = torch.empty((world_size, 2, 2), dtype=torch.float32)
        for destination in range(world_size):
            grad_ag[destination, :, 0] = float(100 * rank + destination)
            grad_ag[destination, :, 1] = source_contributions[rank]
        first = _deterministic_reduce_scatter_sum(grad_ag, group=dist.group.WORLD)
        second = _deterministic_reduce_scatter_sum(grad_ag, group=dist.group.WORLD)
        expected_nonassociative = torch.tensor(source_contributions[0], dtype=torch.float32)
        for source in range(1, world_size):
            expected_nonassociative.add_(torch.tensor(source_contributions[source], dtype=torch.float32))
        expected = torch.empty((2, 2), dtype=torch.float32)
        expected[:, 0] = float(sum(100 * source + rank for source in range(world_size)))
        expected[:, 1] = expected_nonassociative
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_kcp_deterministic_reduce_scatter_routes_then_sums_in_source_order():
    mp.spawn(_run_deterministic_reduce_scatter, args=(4, _free_port()), nprocs=4, join=True)


def _run_participation_token_liveness(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        source = torch.tensor(float(rank + 1), requires_grad=True)
        local_hm = torch.full((1, 1, 1, 2), float(rank), dtype=torch.float32)
        gathered = all_gather_affine_hm(
            local_hm,
            cp_group=dist.group.WORLD,
            cp_size=world_size,
            cp_rank=rank,
            participate=source * 0,
        )
        gathered.sum().backward()
        assert source.grad is not None
        torch.testing.assert_close(source.grad, torch.zeros_like(source), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_kcp_all_gather_participation_token_keeps_every_rank_in_backward():
    mp.spawn(_run_participation_token_liveness, args=(2, _free_port()), nprocs=2, join=True)


def _run_coordinated_preflight_failure(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    original_all_gather = gdn_kcp_module.all_gather_affine_hm
    original_validate = gdn_kcp_module._validate_local_affine_preflight
    try:
        inputs = tuple(tensor.requires_grad_() for tensor in _global_inputs(64))
        plan = compile_gdn_lossless_runtime_plan([64 * world_size], cp_group=dist.group.WORLD, ulysses_size=1)
        observer = make_gdn_cp_runtime_observer("kcp", plan=plan)
        ag_calls = []

        def record_all_gather(*args, **kwargs):
            ag_calls.append(True)
            return original_all_gather(*args, **kwargs)

        gdn_kcp_module.all_gather_affine_hm = record_all_gather

        with pytest.raises(RuntimeError, match="coordinated local-affine preflight failed"):
            resolve_kcp_initial_state(
                *inputs,
                plan=plan,
                cp_group=dist.group.WORLD,
                cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
                use_qk_l2norm=False,
                affine_impl="torch_reference",
                observer=observer,
            )
        snapshot = observer.snapshot()
        ready = [event for event in snapshot.events if event.operation == "kcp_affine_readiness"]
        assert len(ready) == 1 and ready[0].enter == 1 and ready[0].error == 1
        assert not ag_calls

        # A rank-local segment-count error must also be coordinated before AG.
        asymmetric_cu = torch.tensor([0, 32, 64] if rank == 0 else [0, 64], dtype=torch.int32)
        with pytest.raises(RuntimeError, match="coordinated local-affine preflight failed"):
            resolve_kcp_initial_state(
                *inputs,
                plan=plan,
                cp_group=dist.group.WORLD,
                cu_seqlens=asymmetric_cu,
                use_qk_l2norm=False,
                affine_impl="torch_reference",
            )
        assert not ag_calls

        # A first-use TTX forward/VJP warmup error is coordinated too: peers
        # may build their local summary, but no rank can enter affine AG.
        def asymmetric_warmup(*args, **kwargs):
            if rank == 0:
                raise RuntimeError("synthetic TTX warmup failure")

        gdn_kcp_module._validate_local_affine_preflight = asymmetric_warmup
        with pytest.raises(RuntimeError, match="coordinated local-affine preflight failed"):
            resolve_kcp_initial_state(
                *inputs,
                plan=plan,
                cp_group=dist.group.WORLD,
                cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
                use_qk_l2norm=False,
                affine_impl="torch_reference",
            )
        assert not ag_calls
    finally:
        gdn_kcp_module.all_gather_affine_hm = original_all_gather
        gdn_kcp_module._validate_local_affine_preflight = original_validate
        dist.destroy_process_group()


def test_kcp_coordinates_local_affine_failure_before_terminal_rank_enters_ag():
    mp.spawn(_run_coordinated_preflight_failure, args=(2, _free_port()), nprocs=2, join=True)


def _run_per_layer_readiness(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    original_validate = gdn_kcp_module._validate_local_affine_preflight
    original_summary = gdn_kcp_module.local_affine_summary
    original_all_reduce = dist.all_reduce
    readiness_collectives = []
    preflight_calls = []
    try:
        gdn_kcp_module._validate_local_affine_preflight = lambda *args, **kwargs: preflight_calls.append(True)

        def torch_summary(key, value, g, beta, **kwargs):
            return local_affine_summary_fused_torch(
                key,
                value,
                g,
                beta,
                cu_seqlens=kwargs.get("cu_seqlens"),
                use_qk_l2norm=False,
            )

        def record_all_reduce(*args, **kwargs):
            readiness_collectives.append(True)
            return original_all_reduce(*args, **kwargs)

        gdn_kcp_module.local_affine_summary = torch_summary
        dist.all_reduce = record_all_reduce
        observers = []
        layer_ready = {0: False, 1: False}
        coordinate_history = []
        scan_history = []
        for valid_length, layer_id in ((64, 0), (65, 0), (129, 0), (256, 1)):
            plan = compile_gdn_lossless_runtime_plan(
                [valid_length],
                cp_group=dist.group.WORLD,
                ulysses_size=1,
            )
            requires_scan = gdn_kcp_module.kcp_plan_requires_affine_scan(plan)
            coordinate = not layer_ready[layer_id] and requires_scan
            coordinate_history.append(coordinate)
            scan_history.append(requires_scan)
            observer = make_gdn_cp_runtime_observer("kcp", plan=plan)
            observers.append(observer)
            tokens = plan.local.owned_token_count
            torch.manual_seed(1800 + valid_length + rank)
            key = torch.randn(1, tokens, 1, 2, requires_grad=True)
            value = torch.randn(1, tokens, 1, 2, requires_grad=True)
            g = torch.randn(1, tokens, 1, requires_grad=True)
            beta = torch.randn(1, tokens, 1, requires_grad=True)
            resolve_kcp_initial_state(
                key,
                value,
                g,
                beta,
                plan=plan,
                cp_group=dist.group.WORLD,
                cu_seqlens=torch.tensor(plan.local.owned_cu_seqlens, dtype=torch.int32),
                use_qk_l2norm=False,
                affine_impl="ttx_bc8_m1",
                coordinate_readiness=coordinate,
                observer=observer,
            )
            if coordinate:
                layer_ready[layer_id] = True

        # CP4 starts with one owner, then expands to two and three owners.  The
        # first real scan warms every rank exactly once; later owner expansion
        # is hot, and a second layer coordinates independently once.
        assert scan_history == [False, True, True, True]
        assert coordinate_history == [False, True, False, True]
        assert len(readiness_collectives) == 2
        assert len(preflight_calls) == 2
        ready_signatures = []
        for observer in observers:
            events = [event for event in observer.snapshot().events if event.operation == "kcp_affine_readiness"]
            ready_signatures.append(
                (
                    sum(event.enter for event in events),
                    sum(event.exit for event in events),
                    sum(event.error for event in events),
                )
            )
        assert ready_signatures == [(0, 0, 0), (1, 1, 0), (0, 0, 0), (1, 1, 0)]
    finally:
        gdn_kcp_module._validate_local_affine_preflight = original_validate
        gdn_kcp_module.local_affine_summary = original_summary
        dist.all_reduce = original_all_reduce
        dist.destroy_process_group()


def test_kcp_readiness_is_once_per_layer_not_per_dynamic_plan():
    mp.spawn(_run_per_layer_readiness, args=(4, _free_port()), nprocs=4, join=True)


def _run_empty_owner_backward_order(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    original_owned_output_backward = gdn_lossless_module._OwnedToPhysical.backward
    original_physical_input_backward = gdn_lossless_module._PhysicalToOwned.backward
    original_kcp_backward = gdn_kcp_module._KcpAllGatherHm.backward
    try:
        plan = compile_gdn_lossless_runtime_plan([64], cp_group=dist.group.WORLD, ulysses_size=1)
        timeline = []

        def owned_output_backward(ctx, grad_output):
            timeline.append("owned_to_physical")
            return original_owned_output_backward(ctx, grad_output)

        def kcp_backward(ctx, grad_output):
            timeline.append("kcp_reduce_scatter")
            return original_kcp_backward(ctx, grad_output)

        def physical_input_backward(ctx, grad_output):
            timeline.append("physical_to_owned")
            return original_physical_input_backward(ctx, grad_output)

        gdn_lossless_module._OwnedToPhysical.backward = staticmethod(owned_output_backward)
        gdn_kcp_module._KcpAllGatherHm.backward = staticmethod(kcp_backward)
        gdn_lossless_module._PhysicalToOwned.backward = staticmethod(physical_input_backward)
        source_tokens = plan.local.source_token_count
        torch.manual_seed(811 + rank)
        physical_q = torch.randn(1, source_tokens, 1, 2, requires_grad=True)
        physical_k = torch.randn(1, source_tokens, 1, 2, requires_grad=True)
        physical_v = torch.randn(1, source_tokens, 1, 2, requires_grad=True)
        physical_g = torch.randn(1, source_tokens, 1, requires_grad=True)
        physical_beta = torch.randn(1, source_tokens, 1, requires_grad=True)
        owned = tuple(
            physical_to_owned(tensor, plan=plan, cp_group=dist.group.WORLD)
            for tensor in (physical_q, physical_k, physical_v, physical_g, physical_beta)
        )
        query, key, value, g, beta = owned
        key = producer_dtype_l2norm(key)
        cu = torch.tensor(plan.local.owned_cu_seqlens, dtype=torch.int32)
        initial_state = resolve_kcp_initial_state(
            key,
            value,
            g,
            beta,
            plan=plan,
            cp_group=dist.group.WORLD,
            cu_seqlens=cu,
            use_qk_l2norm=False,
            affine_impl="torch_reference",
            extra_participation=make_state_participation(query),
        )
        if plan.local.owned_token_count == 0:
            core = attach_state_dependency(value.new_empty(value.shape), initial_state)
        else:
            core = attach_state_dependency(value + query, initial_state)
        physical_output = owned_to_physical(core, plan=plan, cp_group=dist.group.WORLD)
        physical_output.sum().backward()

        assert timeline.index("owned_to_physical") < timeline.index("kcp_reduce_scatter")
        assert timeline.index("kcp_reduce_scatter") < timeline.index("physical_to_owned")
        assert timeline.count("owned_to_physical") == 1
        assert timeline.count("kcp_reduce_scatter") == 1
        assert timeline.count("physical_to_owned") == 5
        for tensor in (physical_q, physical_k, physical_v, physical_g, physical_beta):
            assert tensor.grad is not None
    finally:
        gdn_lossless_module._OwnedToPhysical.backward = staticmethod(original_owned_output_backward)
        gdn_lossless_module._PhysicalToOwned.backward = staticmethod(original_physical_input_backward)
        gdn_kcp_module._KcpAllGatherHm.backward = staticmethod(original_kcp_backward)
        dist.destroy_process_group()


def test_kcp_empty_owner_preserves_output_ag_and_input_a2a_backward_order():
    mp.spawn(_run_empty_owner_backward_order, args=(2, _free_port()), nprocs=2, join=True)
