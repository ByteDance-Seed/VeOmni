"""Pre-Attention (QKV + A2A) latency sweep across sequence lengths.

Benchmarks the production ``AsyncUlyssesQKVProjection`` fused-signature path
(``qkv_weight=`` from ``FusedQKVLinear``, mirrors patchgen'd qwen3_vl(_moe)
attention forwards) and emits a summary table, raw JSON, and bar chart.

Mode → implementation:

  * ``async``         — ``async_ulysses_qkv_projection(..., dispatch=None)``;
    ``try_resolve_auto`` returns None (bench publishes no active manager),
    falling through to ``_qkv_via_eager`` (production eager path).
  * ``fused``         — same call with ``dispatch=WsPushDispatch(...)``, hitting
    ``_qkv_via_ws_push`` → ``ws_push_forward_impl`` (quack PackQKV GEMM + TMA
    push, then clone+unpad+contiguous finalisation).
  * ``fused_noclone`` — BENCH-ONLY: calls ``ws_push_forward_impl`` directly with
    ``_return_views=True``, skipping the production-mandatory clone+unpad+
    contiguous; outputs are ``peer_out_buf`` views overwritten on the next iter.
    Isolates the 5-10% finalisation tax from raw kernel cost.

Bench injects ``WsPushDispatch`` manually instead of using
``set_active_manager`` + ``try_resolve_auto`` to keep the slot uncontaminated
across tests; ``try_resolve_auto`` overhead is negligible (see
``dev_md/ulysses/PR_doc/Full_intergrate_design.md`` stage ④).

Usage (8x Hopper):

    ULYSSES_WORLD_SIZE=8 pytest tests/parallel/ulysses/test_pre_attn_sweep.py -x -s
"""

import datetime
import gc
import json
import os
import pathlib
import statistics
import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_id, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.async_ulysses import async_ulysses_qkv_projection
from veomni.distributed.sequence_parallel.data import slice_input_tensor
from veomni.models.modules.fused_qkv_linear import FusedQKVLinear
from veomni.ops.kernels.fused_ulysses_projection.ws_push import WsPushDispatch
from veomni.utils.helper import set_seed
from veomni.utils.import_utils import is_torch_npu_available

from .utils import SequenceParallelTest


def _symm_mem_available() -> bool:
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
SEQ_LENS = [32768, 65536, 131072, 262144, 524288]
MODES = ("async", "fused", "fused_noclone")
MODE_LABELS = {
    "async": "Async (Ulysses overlap)",
    "fused": "Fused (ws_push TMA epilogue)",
    "fused_noclone": "Fused (no-clone, BENCH-ONLY ablation)",
}

SWEEP_NUM_WARMUP = int(os.environ.get("SWEEP_NUM_WARMUP", 5))
SWEEP_NUM_ITERS = int(os.environ.get("SWEEP_NUM_ITERS", 30))
SWEEP_NUM_DISCARD = int(os.environ.get("SWEEP_NUM_DISCARD", 3))

# Fixed model shape (gpt-oss-120b attention block)
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = 2880
BATCH_SIZE = 1
DTYPE = torch.bfloat16


_L2_FLUSH_BYTES = 1 << 20


def benchmark_fn(fn, *, warmup, iters, discard, sp_group=None):
    """CUDA-event timer; always L2-flushes, optional per-iter barrier on sp_group.

    Returns dict with keys mean/median/stdev/min/p90/max (milliseconds).
    """
    device = torch.cuda.current_device()
    flush_buf = torch.empty(_L2_FLUSH_BYTES, dtype=torch.int8, device=device)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush_buf.zero_()
        if sp_group is not None:
            dist.barrier(sp_group)
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    samples = samples[discard:] if discard else samples
    assert samples, "benchmark_fn produced no samples after discard"
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": samples[0],
        "p90": samples[max(0, int(len(samples) * 0.9) - 1)],
        "max": samples[-1],
    }


class PreAttentionSweepTest(SequenceParallelTest):
    """Multi-seq-len latency sweep across sync / async / fused pre-attention paths."""

    @property
    def world_size(self):
        return int(os.environ.get("ULYSSES_WORLD_SIZE", 8))

    # ------------------------------------------------------------------
    # Skip / fixture helpers (mirror test_fused_ulysses_pre_attn.py)
    # ------------------------------------------------------------------
    def _maybe_skip_for_fused(self):
        if is_torch_npu_available():
            self.skipTest("npu skip fused ulysses pre-attn sweep")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if torch.cuda.get_device_capability(self.rank)[0] < 9:
            self.skipTest("fused kernel requires sm_90+ (Hopper)")
        if not _symm_mem_available():
            self.skipTest("torch.distributed._symmetric_memory not available")

    @staticmethod
    def _build_fused_qkv_proj(device):
        # Mirrors the production path: ``FusedQKVLinear`` is what patchgen's
        # modify_init installs in qwen3_vl / qwen3_vl_moe attention. The fused
        # weight is broadcast once (vs. three times for separate Linears).
        qkv_proj = FusedQKVLinear(
            hidden_size=HIDDEN_DIM,
            n_q=NUM_Q_HEADS,
            n_kv=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            bias=False,
            device=device,
            dtype=DTYPE,
        )
        dist.broadcast(qkv_proj.weight.data, src=0)
        if qkv_proj.bias is not None:
            dist.broadcast(qkv_proj.bias.data, src=0)
        return qkv_proj

    @staticmethod
    def _init_fused_state(sp_group, device, local_seq, qkv_proj):
        # Local import so non-CUDA hosts can still import this test module.
        from veomni.ops.kernels.fused_ulysses_projection.ws_push import (
            choose_tile_config,
            init_ws_push_state,
        )

        # Delegate to choose_tile_config so bench and prod can't drift on the
        # tile-vs-local_seq threshold; init_ws_push_state enforces
        # epi_tile_M <= local_seq as the final safety net.
        tile_n, pingpong = choose_tile_config(local_seq, dist.get_world_size(sp_group))

        return init_ws_push_state(
            sp_group=sp_group,
            device=device,
            bs=BATCH_SIZE,
            local_seq=local_seq,
            nheads_q=qkv_proj.n_q,
            nheads_k=qkv_proj.n_kv,
            nheads_v=qkv_proj.n_kv,
            head_dim=qkv_proj.head_dim,
            dtype=qkv_proj.weight.dtype,
            tile_n=tile_n,
            pingpong=pingpong,
        )

    @staticmethod
    def _teardown_state(state, sp_group):
        torch.cuda.synchronize()
        dist.barrier(sp_group)
        state.close()

    # ------------------------------------------------------------------
    # Mode dispatcher: build a no_grad pre-attn callable for one (mode, seq_len)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_pre_attn_fn(mode, *, sp_group, part_input, unpad_size, qkv_proj, ws_state):
        if mode in ("async", "fused"):
            # async: dispatch=None → try_resolve_auto returns None (no active
            # manager) → AsyncUlyssesQKVProjection routes to _qkv_via_eager
            # via the fused-signature branch (production eager path).
            # fused : dispatch=WsPushDispatch(...) → _qkv_via_ws_push →
            # ws_push_forward_impl (clone+unpad+contiguous inside).
            # Mirror try_resolve_fused (ws_push.py:305-340): detach so
            # validate_ws_push_dispatch accepts the weight; backward is not
            # exercised by the bench.
            dispatch = (
                WsPushDispatch(state=ws_state, W_qkv_B=qkv_proj.weight.detach(), bias_B=None)
                if mode == "fused"
                else None
            )
            call_kwargs = dict(
                hidden_states=part_input,
                seq_dimension=1,
                head_dimension=2,
                qkv_weight=qkv_proj.weight,
                qkv_bias=qkv_proj.bias,
                n_q=qkv_proj.n_q,
                n_kv=qkv_proj.n_kv,
                norm_type=None,
                norm_q_weight=None,
                norm_q_bias=None,
                norm_k_weight=None,
                norm_k_bias=None,
                normalized_shape=None,
                eps=None,
                unpadded_dim_size=unpad_size,
                head_dim=qkv_proj.head_dim,
                group=sp_group,
                dispatch=dispatch,
            )

            @torch.no_grad()
            def fn():
                return async_ulysses_qkv_projection(**call_kwargs)

            return fn

        if mode == "fused_noclone":
            # Bench-only ablation: bypass AsyncUlyssesQKVProjection entirely
            # and skip _qkv_via_ws_push's clone + unpad + .contiguous(). The
            # returned tensors are zero-copy views into peer_out_buf which
            # the next iter overwrites — only safe because the bench discards
            # outputs. Isolates kernel cost from the production-mandatory
            # finalisation tax (5-10% at long seq).
            from veomni.ops.kernels.fused_ulysses_projection.ws_push import (
                ws_push_forward_impl,
            )

            W_qkv_B = qkv_proj.weight.detach()

            @torch.no_grad()
            def fn():
                return ws_push_forward_impl(
                    part_input,
                    None,
                    ws_state,
                    W_qkv_B=W_qkv_B,
                    bias_B=None,
                    _return_views=True,
                )

            return fn

        raise ValueError(f"Unknown benchmark mode: {mode!r}")

    # ------------------------------------------------------------------
    # Core: benchmark a single (mode, seq_len) pair
    # ``part_input`` / ``unpad_size`` are built once per seq_len by the caller
    # and shared across modes — saves one NCCL broadcast per extra mode and
    # gives modes a fairer per-seq_len input. ``qkv_proj.weight`` is the
    # row-concatenated [q|k|v] Parameter the fused kernel expects; production
    # parity is via patchgen's modify_init building FusedQKVLinear at model
    # init. Host shuffle is intentionally absent — the device-side
    # PackQKVTileScheduler handles NVLink balancing
    # (see .agents/skills/fsdp-shuffle-weight-problem/SKILL.md).
    # ------------------------------------------------------------------
    def _benchmark_single(self, sp_group, mode, *, part_input, unpad_size, local_seq, qkv_proj, device):
        ws_state = None
        if mode in ("fused", "fused_noclone"):
            ws_state = self._init_fused_state(sp_group, device, local_seq, qkv_proj)

        try:
            pre_attn_fn = self._make_pre_attn_fn(
                mode,
                sp_group=sp_group,
                part_input=part_input,
                unpad_size=unpad_size,
                qkv_proj=qkv_proj,
                ws_state=ws_state,
            )

            with torch.no_grad():
                pre_attn_fn()
            torch.cuda.synchronize()

            # async path emits NCCL collectives — per-iter barrier aligns launch
            # boundaries; fused / fused_noclone have no NCCL-visible collectives
            # in-kernel, so the barrier only adds noise.
            stats = benchmark_fn(
                pre_attn_fn,
                warmup=SWEEP_NUM_WARMUP,
                iters=SWEEP_NUM_ITERS,
                discard=SWEEP_NUM_DISCARD,
                sp_group=sp_group if mode == "async" else None,
            )
        finally:
            if ws_state is not None:
                self._teardown_state(ws_state, sp_group)
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier(sp_group)

        return stats

    # ------------------------------------------------------------------
    # Rank-0 summary table
    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary_table(results, ulysses_size):
        seq_lens = sorted(results.keys())
        modes_present = [m for m in MODES if any(results[s].get(m) is not None for s in seq_lens)]
        if not modes_present:
            return

        sl_w = 10
        col_w = 20

        header = f"{'Seq Len':>{sl_w}}"
        for mode in modes_present:
            header += f" | {MODE_LABELS[mode]:^{col_w}}"
        sep = "-" * len(header)

        print(f"\n{sep}")
        print("QKV + A2A  Median Latency Summary  (ms, speedup vs async)")
        print(
            f"  P={ulysses_size}, B={BATCH_SIZE}, Hq={NUM_Q_HEADS}, "
            f"Hkv={NUM_KV_HEADS}, D={HEAD_DIM}, hidden={HIDDEN_DIM}"
        )
        print(sep)
        print(header)
        print(sep)

        for seq_len in seq_lens:
            label = f"{seq_len // 1024}K"
            row = f"{label:>{sl_w}}"
            # Baseline = async (the existing eager Ulysses path users run in
            # production); fused / fused_noclone are compared against it.
            async_median = results[seq_len].get("async", {}).get("median") if results[seq_len].get("async") else None
            for mode in modes_present:
                stats = results[seq_len].get(mode)
                if stats is None:
                    cell = "N/A"
                else:
                    median = stats["median"]
                    if async_median and async_median > 0 and median > 0:
                        speedup = async_median / median
                        cell = f"{median:7.3f} ms (x{speedup:.2f})"
                    else:
                        cell = f"{median:7.3f} ms"
                row += f" | {cell:^{col_w}}"
            print(row)

        print(sep)

    # ------------------------------------------------------------------
    # Rank-0 JSON dump
    # ------------------------------------------------------------------
    @staticmethod
    def _save_results_json(results, ulysses_size):
        output_dir = pathlib.Path(__file__).resolve().parent / "output"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pre_attn_sweep_P{ulysses_size}_{timestamp}.json"

        serializable = {
            str(seq_len): {mode: stats for mode, stats in modes.items() if stats is not None}
            for seq_len, modes in results.items()
        }

        output = {
            "metadata": {
                "world_size": ulysses_size,
                "num_q_heads": NUM_Q_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "hidden_dim": HIDDEN_DIM,
                "batch_size": BATCH_SIZE,
                "num_warmup": SWEEP_NUM_WARMUP,
                "num_iters": SWEEP_NUM_ITERS,
                "num_discard": SWEEP_NUM_DISCARD,
                "timestamp": timestamp,
            },
            "results": serializable,
        }

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Rank-0 normalised bar chart
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_chart(results, ulysses_size):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available - skipping chart generation")
            return None

        seq_lens = sorted(results.keys())
        modes_present = [m for m in MODES if all(results[s].get(m) is not None for s in seq_lens)]
        if not modes_present or "async" not in modes_present:
            print("async mode missing - cannot normalise against async baseline, skipping chart")
            return None

        normalised = {mode: [] for mode in modes_present}
        raw_medians = {mode: [] for mode in modes_present}

        for seq_len in seq_lens:
            async_median = results[seq_len]["async"]["median"]
            for mode in modes_present:
                median = results[seq_len][mode]["median"]
                normalised[mode].append(median / async_median if async_median > 0 else 0)
                raw_medians[mode].append(median)

        x = np.arange(len(seq_lens))
        n_modes = len(modes_present)
        width = 0.22
        offsets = np.linspace(-(n_modes - 1) * width / 2, (n_modes - 1) * width / 2, n_modes)

        colors = {
            "async": "#55A868",
            "fused": "#C44E52",
            "fused_noclone": "#8172B2",
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, mode in enumerate(modes_present):
            bars = ax.bar(
                x + offsets[i],
                normalised[mode],
                width,
                label=MODE_LABELS[mode],
                color=colors.get(mode, "#999999"),
                edgecolor="white",
            )
            for j, bar in enumerate(bars):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{raw_medians[mode][j]:.2f} ms",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Normalised Latency (async = 1.0, lower is better)")
        ax.set_title(
            f"Pre-Attention (QKV + A2A) Latency Sweep  "
            f"(B={BATCH_SIZE}, Hq={NUM_Q_HEADS}, Hkv={NUM_KV_HEADS}, "
            f"D={HEAD_DIM}, hidden={HIDDEN_DIM}, P={ulysses_size})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s // 1024}K" for s in seq_lens])
        ax.legend()
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        output_dir = pathlib.Path(__file__).resolve().parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = output_dir / f"pre_attn_sweep_P{ulysses_size}_{timestamp}.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved to {chart_path}")
        return chart_path

    # ------------------------------------------------------------------
    # Test entry point
    # ------------------------------------------------------------------
    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip fused ulysses pre-attn sweep")
    def test_pre_attn_seq_len_sweep(self):
        sp_group = self._get_process_group()
        ulysses_size = dist.get_world_size(sp_group)
        device = torch.device(get_device_id())
        torch.manual_seed(0xC0FFEE)

        fused_available = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.rank)[0] >= 9
            and _symm_mem_available()
            and not is_torch_npu_available()
        )

        qkv_proj = self._build_fused_qkv_proj(device)

        results = {}

        if self.rank == 0:
            n_samples = SWEEP_NUM_ITERS - SWEEP_NUM_DISCARD
            print(
                f"\n{'=' * 90}\n"
                f"Pre-Attention Sweep Benchmark (QKV + A2A, forward only)\n"
                f"  B={BATCH_SIZE}, Hq={NUM_Q_HEADS}, Hkv={NUM_KV_HEADS}, "
                f"D={HEAD_DIM}, hidden={HIDDEN_DIM}, P={ulysses_size}\n"
                f"  Warmup={SWEEP_NUM_WARMUP}, Iters={SWEEP_NUM_ITERS}, "
                f"Discard={SWEEP_NUM_DISCARD} (effective samples={n_samples})\n"
                f"  Seq lengths: {SEQ_LENS}\n"
                f"  Fused mode: {'available' if fused_available else 'SKIPPED (requires SM90+ and symm-mem)'}\n"
                f"{'=' * 90}"
            )

        for seq_len in SEQ_LENS:
            if seq_len % ulysses_size != 0:
                if self.rank == 0:
                    print(f"\n--- seq_len = {seq_len} skipped (not divisible by world_size={ulysses_size}) ---")
                continue

            results[seq_len] = {}
            if self.rank == 0:
                print(f"\n--- seq_len = {seq_len} ({seq_len // 1024}K) ---")

            # Build input once per seq_len; all modes share it so they are
            # benchmarked against bit-identical data and we save one NCCL
            # broadcast per extra mode.
            local_seq = seq_len // ulysses_size
            full_input = torch.randn(BATCH_SIZE, seq_len, HIDDEN_DIM, device=device, dtype=DTYPE)
            dist.broadcast(full_input, src=0)
            part_input = slice_input_tensor(full_input, dim=1, group=sp_group).contiguous()
            unpad_size = full_input.size(1)
            del full_input

            try:
                for mode in MODES:
                    if mode in ("fused", "fused_noclone") and not fused_available:
                        continue
                    if mode == "async" and is_torch_npu_available():
                        continue

                    stats = self._benchmark_single(
                        sp_group,
                        mode,
                        part_input=part_input,
                        unpad_size=unpad_size,
                        local_seq=local_seq,
                        qkv_proj=qkv_proj,
                        device=device,
                    )
                    results[seq_len][mode] = stats

                    if self.rank == 0:
                        label = MODE_LABELS[mode]
                        print(
                            f"  {label}  QKV+A2A: "
                            f"median {stats['median']:8.3f} ms  |  "
                            f"mean {stats['mean']:8.3f} ms  |  "
                            f"stdev {stats['stdev']:7.3f} ms  |  "
                            f"min {stats['min']:8.3f} ms  |  "
                            f"p90 {stats['p90']:8.3f} ms"
                        )
            finally:
                del part_input
                gc.collect()
                torch.cuda.empty_cache()

        if self.rank == 0:
            self._print_summary_table(results, ulysses_size)
            print(f"\n{'=' * 90}")
            self._save_results_json(results, ulysses_size)
            self._generate_chart(results, ulysses_size)
            print(f"{'=' * 90}")


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    run_tests()
