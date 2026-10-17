"""Opt-in HCCL replay against an exact baseline checkout, before full training.

Run with torchrun on 32 NPUs and QWEN4_EXP_PACKED_BASELINE pointing to the
public #1275 baseline 9287166bbc00e741a89a87b499169c8fe3bc94b8.
The candidate must also include #1275's NPU-safe TopK correction.
This measures the selector, not full-model throughput. No optimizer is used.
"""

import hashlib
import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from veomni.models.transformers.qwen4_exp import packed_utils
from veomni.ops.kernels.qwen4_exp.npu_qsa import qsa_attn_npu_fused


def load_baseline():
    root = Path(os.environ["QWEN4_EXP_PACKED_BASELINE"])
    path = root / "veomni/models/transformers/qwen4_exp/packed_utils.py"
    # Exact public #1275 baseline file identities (9287166bbc00e741a89a87b499169c8fe3bc94b8).
    assert (
        hashlib.sha256((root / "veomni/models/transformers/qwen4_exp/packed_utils.py").read_bytes()).hexdigest()
        == "cf410f1024f56f3e890ccebc4511e209d1bc0a179cccadcc4bb76bd7de7a72d8"
    )
    assert (
        hashlib.sha256((root / "veomni/distributed/context_parallel/dsa_cp.py").read_bytes()).hexdigest()
        == "0317c3ba88f75a25bdbde8265c7f0d13c84e2bc6d5b8d13a1391981fe83a3684"
    )
    spec = importlib.util.spec_from_file_location("veomni.models.transformers.qwen4_exp._baseline_packed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Preserve the baseline's device-count readback for an honest timing comparison.
    source = (root / "veomni/distributed/context_parallel/dsa_cp.py").read_text()
    namespace = {
        "__name__": "veomni.distributed.context_parallel._baseline_dsa",
        "__package__": "veomni.distributed.context_parallel",
    }
    exec(compile(source, str(root / "veomni/distributed/context_parallel/dsa_cp.py"), "exec"), namespace)
    module.all_gather_compressed_rows = namespace["all_gather_compressed_rows"]
    return module


def rotary(q, cos, sin, unsqueeze_dim=1):
    a, b = q.chunk(2, dim=-1)
    return q * cos.unsqueeze(unsqueeze_dim) + torch.cat((-b, a), dim=-1) * sin.unsqueeze(unsqueeze_dim)


def replay(baseline, group, rank, world, lengths, *, batch=1, real_shape=False):
    total = sum(lengths) // batch
    local = total // world
    hidden_size = 2560 if real_shape else 32
    dim = 128 if real_shape else 16
    budget = 2048 if real_shape else 32
    indexer = SimpleNamespace(
        index_kv_heads=1,
        index_n_heads=4,
        index_head_dim=dim,
        compress_ratio=4,
        block_topk=budget // 4,
        token_budget=budget,
        index_qk_proj=torch.nn.Linear(hidden_size, 5 * dim, bias=False).npu().bfloat16(),
        q_layernorm=torch.nn.RMSNorm(dim).npu().bfloat16(),
        k_layernorm=torch.nn.RMSNorm(dim).npu().bfloat16(),
    )
    # All ranks use the same projection; hidden shards differ by rank.
    for layer in (indexer.index_qk_proj, indexer.q_layernorm, indexer.k_layernorm):
        for param in layer.parameters():
            dist.broadcast(param.data, dist.get_global_rank(group, 0), group=group)
            param.requires_grad_(False)
    hidden = torch.randn(batch, local, hidden_size, device="npu", dtype=torch.bfloat16)
    positions = (torch.ones(batch, total, dim, device="npu"), torch.zeros(batch, total, dim, device="npu"))
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int32)
    common = dict(group=group, rank=rank, world_size=world, apply_rotary_pos_emb=rotary)
    cache = {}

    def old():
        return baseline.compact_qsa_select(indexer, hidden, positions, cu, **common)

    def new():
        return packed_utils.compact_qsa_select(
            indexer, hidden, positions, cu, packed_seq_lens=lengths, layout_cache=cache, **common
        )

    with torch.no_grad():
        torch.npu.synchronize()
        start = time.perf_counter()
        expected = old()
        torch.npu.synchronize()
        baseline_cold = time.perf_counter() - start
        start = time.perf_counter()
        actual = new()
        torch.npu.synchronize()
        cached_cold = time.perf_counter() - start
        assert torch.equal(expected, actual), "Selector indices changed"
        times = {"baseline": [], "cached": []}
        for repeat in range(3):
            # Alternate order to avoid always favoring the warmed candidate.
            for label, call in (
                (("baseline", old), ("cached", new)) if repeat % 2 == 0 else (("cached", new), ("baseline", old))
            ):
                dist.barrier(group=group)
                torch.npu.synchronize()
                start = time.perf_counter()
                call()
                torch.npu.synchronize()
                times[label].append(time.perf_counter() - start)
    if not real_shape:
        q = torch.randn(batch, 4, local, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch, 1, total, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        output0 = qsa_attn_npu_fused(q, k, v, expected, query_chunk_size=64)
        output1 = qsa_attn_npu_fused(q, k, v, actual, query_chunk_size=64)
        grad = torch.randn_like(output0)
        grads0 = torch.autograd.grad(output0, (q, k, v), grad)
        grads1 = torch.autograd.grad(output1, (q, k, v), grad)
        torch.testing.assert_close(output0, output1, rtol=0, atol=0)
        for x, y in zip(grads0, grads1):
            torch.testing.assert_close(x, y, rtol=1e-3, atol=1e-3)
    metrics = torch.tensor([sum(times["baseline"]) / 3, sum(times["cached"]) / 3], device="npu")
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        print(
            "PACKED_NATIVE_REPLAY "
            + json.dumps(
                {
                    "sp": world,
                    "batch": batch,
                    "global_length": total,
                    "real_shape": real_shape,
                    "indices_exact": True,
                    "fwd_bwd_checked": not real_shape,
                    "max_rank_warm_seconds": metrics.cpu().tolist(),
                    "owner0_cold_seconds": [baseline_cold, cached_cold],
                    "cache_entries": len(cache),
                }
            ),
            flush=True,
        )


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=15))
    assert dist.get_world_size() == 32
    torch.manual_seed(1229)
    baseline = load_baseline()
    for size in (16, 32):
        groups = [dist.new_group(list(range(start, start + size))) for start in range(0, 32, size)]
        group = groups[dist.get_rank() // size]
        rank = dist.get_rank() % size
        replay(baseline, group, rank, size, (3, 258, 763))
        replay(baseline, group, rank, size, (5, 507, 3, 509), batch=2)
        replay(baseline, group, rank, size, (1,) * 1024)
        replay(baseline, group, rank, size, (32771, 65537, 32764), real_shape=True)
        dist.barrier()
    print(f"PACKED_NATIVE_PASS rank={dist.get_rank()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
