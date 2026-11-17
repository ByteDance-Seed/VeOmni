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

"""Microbenchmark for grouped Q/K heads in FlashQLA GatedDeltaNet.

Run on an SM90/SM100 GPU with ``flash_qla`` installed:
    python tests/ops/benchmark_gdn_grouped_heads.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.configuration_base import get_veomni_config_class
from veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu import Qwen3_5GatedDeltaNet
from veomni.ops import resolve_op
from veomni.utils.import_utils import is_package_available


@dataclass
class BenchResult:
    name: str
    total_tokens: int
    k_heads: int
    v_heads: int
    head_dim: int
    fwd_ms: float
    bwd_ms: float
    total_ms: float
    peak_mem_mb: float
    saved_repeat_mb: float


def _time_cuda(fn, *, warmup: int, iters: int) -> tuple[float, float, float, float]:
    for _ in range(warmup):
        out, grad_out = fn()
        out.backward(grad_out)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    out, grad_out = fn()
    out.backward(grad_out)
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fwd_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    bwd_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]

    for i in range(iters):
        f_start, f_end = fwd_events[i]
        b_start, b_end = bwd_events[i]
        f_start.record()
        out, grad_out = fn()
        f_end.record()
        b_start.record()
        out.backward(grad_out)
        b_end.record()

    torch.cuda.synchronize()
    fwd_ms = sum(s.elapsed_time(e) for s, e in fwd_events) / iters
    bwd_ms = sum(s.elapsed_time(e) for s, e in bwd_events) / iters
    return fwd_ms, bwd_ms, fwd_ms + bwd_ms, peak_mem_mb


def bench_kernel(
    *,
    total_tokens: int,
    num_seqs: int,
    k_heads: int,
    v_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
) -> tuple[BenchResult, BenchResult]:
    wrapper = resolve_op("chunk_gated_delta_rule", "standard", "flash_qla").wrapper
    group_size = v_heads // k_heads
    seq_len = total_tokens // num_seqs
    cu_seqlens = torch.arange(0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32)

    q_base = torch.randn(1, total_tokens, k_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_base = torch.randn(1, total_tokens, k_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_base = torch.randn(1, total_tokens, v_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    g_base = -torch.rand(1, total_tokens, v_heads, device="cuda", dtype=torch.float32)
    beta_base = torch.sigmoid(torch.randn(1, total_tokens, v_heads, device="cuda", dtype=torch.bfloat16))
    grad_out = torch.randn(1, total_tokens, v_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    repeat_bytes = 2 * total_tokens * (v_heads - k_heads) * head_dim * 2
    saved_repeat_mb = repeat_bytes / (1024 * 1024)

    def run_expanded():
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        g = g_base.detach().clone().requires_grad_(True)
        beta = beta_base.detach().clone().requires_grad_(True)
        q_rep = q.repeat_interleave(group_size, dim=2)
        k_rep = k.repeat_interleave(group_size, dim=2)
        out, _ = wrapper(
            q_rep,
            k_rep,
            v,
            g,
            beta,
            None,
            cu_seqlens,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return out, grad_out

    def run_grouped():
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        g = g_base.detach().clone().requires_grad_(True)
        beta = beta_base.detach().clone().requires_grad_(True)
        out, _ = wrapper(
            q,
            k,
            v,
            g,
            beta,
            None,
            cu_seqlens,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return out, grad_out

    exp_fwd, exp_bwd, exp_tot, exp_mem = _time_cuda(run_expanded, warmup=warmup, iters=iters)
    grp_fwd, grp_bwd, grp_tot, grp_mem = _time_cuda(run_grouped, warmup=warmup, iters=iters)
    return (
        BenchResult(
            "kernel_expanded_qk", total_tokens, k_heads, v_heads, head_dim, exp_fwd, exp_bwd, exp_tot, exp_mem, 0.0
        ),
        BenchResult(
            "kernel_grouped_qk",
            total_tokens,
            k_heads,
            v_heads,
            head_dim,
            grp_fwd,
            grp_bwd,
            grp_tot,
            grp_mem,
            saved_repeat_mb,
        ),
    )


def bench_layer(
    *,
    total_tokens: int,
    num_seqs: int,
    k_heads: int,
    v_heads: int,
    head_dim: int,
    hidden_size: int,
    warmup: int,
    iters: int,
) -> tuple[BenchResult, BenchResult]:
    ops_cfg = OpsImplementationConfig(
        rms_norm_gated_implementation="fla",
        causal_conv1d_implementation="fla",
        chunk_gated_delta_rule_implementation="flash_qla",
    )
    config_class = get_veomni_config_class(Qwen3_5TextConfig)
    config = config_class(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=k_heads,
        linear_num_value_heads=v_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        ops_implementation=ops_cfg,
    )
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)

    seq_len = total_tokens // num_seqs
    cu_seqlens = torch.arange(0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32)
    x_base = torch.randn(1, total_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn_like(x_base)
    repeat_bytes = 2 * total_tokens * (v_heads - k_heads) * head_dim * 2
    saved_repeat_mb = repeat_bytes / (1024 * 1024)

    flash_qla_op = layer.veomni_chunk_gated_delta_rule

    class _ExpandedFlashQlaProxy:
        impl = "fla"

        def __call__(self, *args, **kwargs):
            return flash_qla_op(*args, **kwargs)

    def run_layer(grouped: bool):
        layer.veomni_chunk_gated_delta_rule = flash_qla_op if grouped else _ExpandedFlashQlaProxy()
        layer.zero_grad(set_to_none=True)
        x = x_base.detach().clone().requires_grad_(True)
        out = layer(x, cu_seq_lens_q=cu_seqlens)
        return out, grad_out

    exp_fwd, exp_bwd, exp_tot, exp_mem = _time_cuda(lambda: run_layer(False), warmup=warmup, iters=iters)
    grp_fwd, grp_bwd, grp_tot, grp_mem = _time_cuda(lambda: run_layer(True), warmup=warmup, iters=iters)
    return (
        BenchResult(
            "layer_expanded_qk", total_tokens, k_heads, v_heads, head_dim, exp_fwd, exp_bwd, exp_tot, exp_mem, 0.0
        ),
        BenchResult(
            "layer_grouped_qk",
            total_tokens,
            k_heads,
            v_heads,
            head_dim,
            grp_fwd,
            grp_bwd,
            grp_tot,
            grp_mem,
            saved_repeat_mb,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not is_package_available("flash_qla"):
        raise RuntimeError("benchmark_gdn_grouped_heads requires CUDA and flash_qla")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    configs = [
        (4096, 2, 4, 16, 128, 2048),
        (8192, 4, 4, 16, 128, 2048),
        (8192, 4, 8, 16, 128, 2048),
    ]
    print(
        f"{'mode':<20} {'tokens':>7} {'Hk->Hv':>8} {'D':>5} "
        f"{'fwd(ms)':>9} {'bwd(ms)':>9} {'total(ms)':>10} {'speedup':>9} {'peak_MB':>10} {'saved_Q/K_MB':>13}"
    )
    print("-" * 108)
    for total_tokens, num_seqs, k_heads, v_heads, head_dim, hidden_size in configs:
        for exp_res, grp_res in (
            bench_kernel(
                total_tokens=total_tokens,
                num_seqs=num_seqs,
                k_heads=k_heads,
                v_heads=v_heads,
                head_dim=head_dim,
                warmup=args.warmup,
                iters=args.iters,
            ),
            bench_layer(
                total_tokens=total_tokens,
                num_seqs=num_seqs,
                k_heads=k_heads,
                v_heads=v_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                warmup=args.warmup,
                iters=args.iters,
            ),
        ):
            speedup = (exp_res.total_ms - grp_res.total_ms) / exp_res.total_ms * 100.0
            for res, sp in ((exp_res, 0.0), (grp_res, speedup)):
                print(
                    f"{res.name:<20} {res.total_tokens:>7} {f'{res.k_heads}->{res.v_heads}':>8} {res.head_dim:>5} "
                    f"{res.fwd_ms:>9.3f} {res.bwd_ms:>9.3f} {res.total_ms:>10.3f} "
                    f"{sp:>+8.2f}% {res.peak_mem_mb:>10.1f} {res.saved_repeat_mb:>13.1f}"
                )


if __name__ == "__main__":
    main()
