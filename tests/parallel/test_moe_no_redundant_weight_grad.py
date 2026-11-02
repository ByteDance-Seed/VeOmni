"""Affected owner suffixes are computed once after ordered prefix receipt."""

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.parallel.test_moe_ep_load_balance_planner import _selected_rows
from tests.parallel.test_moe_ep_load_balance_planner import ep_load_balance as ep


def _worker(rank, rendezvous):
    spec = importlib.util.spec_from_file_location("ep_grad_selection_test", Path(ep.__file__).with_name("ep_grad.py"))
    grad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grad)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        counts = torch.tensor([[70, 2, 90, 5, 0, 7, 0, 3]] * 4, dtype=torch.long)
        selected = _selected_rows(counts.tolist())[rank]
        for slots in (1, 2):
            plan = ep._build_ep_balance_plan_from_counts(selected, counts, 8, rank, slots, token_granularity=32)
            assert plan.replicas
            torch.manual_seed(73)
            full_a = [torch.randn(int(total), 3).bfloat16() for total in counts.sum(0)]
            full_b = [torch.randn(int(total), 5).bfloat16() for total in counts.sum(0)]
            sizes = plan.tokens_per_local_physical_expert.sum(0)
            pieces_a = [full_a[0][:0] for _ in sizes]
            pieces_b = [full_b[0][:0] for _ in sizes]
            cursors = [0] * 8
            for replica in plan.replicas:
                expert = replica.source_expert
                start, end = cursors[expert], cursors[expert] + replica.moved_tokens
                if replica.target_rank == rank:
                    pieces_a[2 + replica.target_slot] = full_a[expert][start:end]
                    pieces_b[2 + replica.target_slot] = full_b[expert][start:end]
                cursors[expert] = end
            for local in range(2):
                expert = rank * 2 + local
                pieces_a[local] = full_a[expert][cursors[expert] :]
                pieces_b[local] = full_b[expert][cursors[expert] :]
            assert [len(piece) for piece in pieces_a] == sizes.tolist()
            a, b = torch.cat(pieces_a), torch.cat(pieces_b)
            c = torch.full((len(sizes), 3, 5), float("nan"))
            calls = []

            def reference(aa, bb, seed=None):
                acc = torch.zeros(3, 5) if seed is None else seed.clone()
                for start in range(0, len(aa), 32):
                    acc.add_(aa[start : start + 32].float().T @ bb[start : start + 32].float())
                return acc

            def launcher(aa, bb, cc, ends, original_total, accumulate, *, group_indices=None, parent=c, records=calls):
                offset = (cc.data_ptr() - parent.data_ptr()) // (15 * parent.element_size())
                active = list(range(len(ends))) if group_indices is None else group_indices.tolist()
                endpoints = [0, *ends.tolist()]
                for group in active:
                    records.append((offset + group, accumulate, endpoints[group + 1] - endpoints[group]))
                    cc[group].copy_(
                        reference(
                            aa[endpoints[group] : endpoints[group + 1]],
                            bb[endpoints[group] : endpoints[group + 1]],
                            cc[group] if accumulate else None,
                        )
                    )

            run = grad.make_ordered_weight_grad(plan, dist.group.WORLD, torch.device("cpu"), launcher=launcher)
            run(a, b, c, sizes.cumsum(0), len(a), transpose_a=True)
            expected_sync = c[:2].clone()
            dist.barrier()
            c.fill_(float("nan"))
            calls.clear()
            pending = run.start(a, b, c, sizes.cumsum(0), len(a), transpose_a=True)
            # Independent local work is legal before waiting for prefix data.
            independent = a.float().square().sum()
            assert torch.isfinite(independent)
            pending.wait()
            pending.wait()
            assert torch.equal(c[:2], expected_sync)
            affected = {r.source_local_expert for r in plan.replicas if r.source_rank == rank}
            for local in range(2):
                expert = rank * 2 + local
                assert torch.equal(c[local], reference(full_a[expert], full_b[expert]))
                local_calls = [call for call in calls if call[0] == local]
                if local in affected:
                    assert len(local_calls) == int(sizes[local] > 0)
                    assert all(call[1] for call in local_calls)
                else:
                    assert len(local_calls) == 1 and not local_calls[0][1]
    finally:
        dist.destroy_process_group()


def test_owner_suffix_is_not_computed_before_prefix_arrives(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'selected_wgrad'}",), nprocs=4, join=True)
