#!/usr/bin/env python3
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

"""Compare the two-rank NPU Ring-attention VJP with monolithic fusion attention."""

from __future__ import annotations

import argparse
import json
import os


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--nl2-limit", type=float, default=2e-2)
    parser.add_argument("--bad-frac-limit", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def main():  # noqa: C901 - an executable diagnostic is clearest as one closed routine
    args = _args()

    import torch
    import torch.distributed as dist
    import torch_npu

    from veomni.distributed.context_parallel.packed_sharding import (
        apply_packed_context_parallel_partition,
        build_packed_context_parallel_partition,
    )
    from veomni.distributed.context_parallel.ring_attention import ringattn_context_parallel
    from veomni.distributed.context_parallel.sharding import balanced_cp_slice

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise RuntimeError(f"gate requires WORLD_SIZE=2, got {world}")
    if args.seq_len % 16 or args.q_heads % args.kv_heads:
        raise ValueError("seq_len must be divisible by 16 and q_heads by kv_heads")
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    device = torch.device(f"npu:{local_rank}")
    scale = args.head_dim**-0.5
    mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu_(1)

    def fixed(shape, stream):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + stream)
        return (torch.randn(shape, generator=generator) * 0.25).to(torch.bfloat16).to(device)

    query = fixed((1, args.q_heads, args.seq_len, args.head_dim), 1)
    key = fixed((1, args.kv_heads, args.seq_len, args.head_dim), 2)
    value = fixed((1, args.kv_heads, args.seq_len, args.head_dim), 3)
    dout = fixed((1, args.q_heads, args.seq_len, args.head_dim), 4)

    def raw_fusion(q, k, v, dy):
        length = q.size(2)
        forward = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            args.q_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            scale=scale,
            pre_tockens=length,
            next_tockens=0,
            keep_prob=1.0,
            sparse_mode=3,
        )
        backward = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            dy,
            args.q_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=mask,
            softmax_max=forward[1],
            softmax_sum=forward[2],
            attention_in=forward[0],
            scale_value=scale,
            pre_tockens=length,
            next_tockens=0,
            sparse_mode=3,
            keep_prob=1.0,
        )
        return forward[0], backward[0], backward[1], backward[2]

    def reference(cu=None):
        if cu is None:
            return raw_fusion(query, key, value, dout)
        pieces = [[], [], [], []]
        for start, end in zip(cu[:-1], cu[1:]):
            got = raw_fusion(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                dout[:, :, start:end],
            )
            for destination, tensor in zip(pieces, got):
                destination.append(tensor)
        return tuple(torch.cat(group, dim=2) for group in pieces)

    def metrics(actual, expected):
        actual_float, expected_float = actual.float(), expected.float()
        difference = (actual_float - expected_float).abs()
        finite = bool(torch.isfinite(actual_float).all().item() and torch.isfinite(expected_float).all().item())
        bad = difference > (args.atol + args.rtol * expected_float.abs())
        return {
            "finite": finite,
            "max_abs": float(difference.max().item()),
            "nl2": float((difference.norm() / expected_float.norm().clamp_min(1e-12)).item()),
            "bad_frac": float(bad.float().mean().item()),
            "ref_abs_max": float(expected_float.abs().max().item()),
        }

    def run_case(name, cu=None):
        expected = reference(cu)
        if cu is None:
            local = tuple(balanced_cp_slice(tensor, world, rank, dim=2) for tensor in (query, key, value, dout))
            local_expected = tuple(balanced_cp_slice(tensor, world, rank, dim=2) for tensor in expected)
            local_cu = None
        else:
            cu_tensor = torch.tensor(cu, dtype=torch.int32)
            partition = build_packed_context_parallel_partition(cu_tensor, cp_size=world, cp_rank=rank)
            local = tuple(
                apply_packed_context_parallel_partition(tensor, partition, dim=2)
                for tensor in (query, key, value, dout)
            )
            local_expected = tuple(
                apply_packed_context_parallel_partition(tensor, partition, dim=2) for tensor in expected
            )
            local_cu = partition.local_cu_seqlens

        local_query, local_key, local_value = (tensor.detach().requires_grad_(True) for tensor in local[:3])
        output = ringattn_context_parallel(
            local_query,
            local_key,
            local_value,
            args.q_heads,
            dist.group.WORLD,
            list(range(world)),
            softmax_scale=scale,
            backend="npu",
            cu_seqlens=local_cu,
        )
        torch.autograd.backward(output, local[3])
        torch.npu.synchronize()
        report = {
            metric_name: metrics(actual, reference_tensor)
            for metric_name, actual, reference_tensor in zip(
                ("out", "dq", "dk", "dv"),
                (output.detach(), local_query.grad, local_key.grad, local_value.grad),
                local_expected,
            )
        }
        local_ok = all(
            item["finite"] and item["nl2"] <= args.nl2_limit and item["bad_frac"] <= args.bad_frac_limit
            for item in report.values()
        )
        ok = torch.tensor([int(local_ok)], dtype=torch.int32, device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        print(
            "AI4SE_RING_NPU_VJP_CASE",
            json.dumps({"case": name, "rank": rank, "ok": bool(ok.item()), "metrics": report}, sort_keys=True),
            flush=True,
        )
        if not ok.item():
            raise AssertionError(f"{name} NPU Ring VJP parity failed")

    try:
        run_case("dense")
        quarter = args.seq_len // 4
        run_case("packed", [0, quarter, 2 * quarter, args.seq_len])
        if rank == 0:
            print(
                "AI4SE_RING_NPU_VJP_GATE_OK",
                json.dumps(
                    {
                        "world_size": world,
                        "seq_len": args.seq_len,
                        "q_heads": args.q_heads,
                        "kv_heads": args.kv_heads,
                        "head_dim": args.head_dim,
                        "torch": torch.__version__,
                        "torch_npu": torch_npu.__version__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
