"""Diagnostic numerical-difference test: ws_push fused QKV-proj vs eager async-ulysses.

Coverage not duplicated by test_fused_ulysses_pre_attn.py / test_fused_ulysses_backward.py:
  * per-tensor max/mean/rel diagnostics for noise-regression visibility
  * unpadded_dim_size % world_size != 0 (seq-not-divisible padding path)
  * buffer-alias safety: a second fused forward must not clobber the first cloned output
"""

import os
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
from veomni.ops.kernels.fused_ulysses_projection.ws_push import WsPushDispatch
from veomni.utils.import_utils import is_torch_npu_available

from .utils import SequenceParallelTest


def _symm_mem_available() -> bool:
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        return True
    except ImportError:
        return False


class FusedVsEagerNumericalTest(SequenceParallelTest):
    BS = 2
    HEAD_DIM = 64
    DTYPE = torch.bfloat16
    ATOL = 5e-2
    RTOL = 5e-2

    @property
    def world_size(self):
        return int(os.environ.get("VEOMNI_TEST_WORLD_SIZE", "2"))

    def _maybe_skip_for_fused(self):
        if is_torch_npu_available():
            self.skipTest("npu skip fused-vs-eager numerical")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if torch.cuda.get_device_capability(self.rank)[0] < 9:
            self.skipTest("fused kernel requires sm_90+ (Hopper)")
        if not _symm_mem_available():
            self.skipTest("torch.distributed._symmetric_memory not available")

    @classmethod
    def _build_inputs(cls, sp_group, device, *, local_seq, nheads_q, nheads_kv, with_bias):
        hidden_dim = nheads_q * cls.HEAD_DIM
        seq_global = local_seq * dist.get_world_size(sp_group)

        full_x = torch.randn(cls.BS, seq_global, hidden_dim, dtype=cls.DTYPE, device=device)
        dist.broadcast(full_x, src=0)
        local_x = slice_input_tensor(full_x, dim=1, group=sp_group).contiguous()

        q_w = torch.randn(nheads_q * cls.HEAD_DIM, hidden_dim, dtype=cls.DTYPE, device=device)
        k_w = torch.randn(nheads_kv * cls.HEAD_DIM, hidden_dim, dtype=cls.DTYPE, device=device)
        v_w = torch.randn(nheads_kv * cls.HEAD_DIM, hidden_dim, dtype=cls.DTYPE, device=device)
        for t in (q_w, k_w, v_w):
            dist.broadcast(t, src=0)

        if with_bias:
            q_b = torch.randn(nheads_q * cls.HEAD_DIM, dtype=cls.DTYPE, device=device)
            k_b = torch.randn(nheads_kv * cls.HEAD_DIM, dtype=cls.DTYPE, device=device)
            v_b = torch.randn(nheads_kv * cls.HEAD_DIM, dtype=cls.DTYPE, device=device)
            for t in (q_b, k_b, v_b):
                dist.broadcast(t, src=0)
        else:
            q_b = k_b = v_b = None
        return local_x, q_w, k_w, v_w, q_b, k_b, v_b

    @staticmethod
    def _clone_with_grad(t):
        if t is None:
            return None
        return t.clone().detach().requires_grad_(True)

    @classmethod
    def _init_state(cls, sp_group, device, *, local_seq, nheads_q, nheads_kv):
        from veomni.ops.kernels.fused_ulysses_projection.ws_push import init_ws_push_state

        # pingpong=True forces epi_tile_m=64 under quack 0.3.4 so each PackQKV epilogue
        # subtile maps to one batch; without it odd-indexed batches receive no TMA write.
        return init_ws_push_state(
            sp_group=sp_group,
            device=device,
            bs=cls.BS,
            local_seq=local_seq,
            nheads_q=nheads_q,
            nheads_k=nheads_kv,
            nheads_v=nheads_kv,
            head_dim=cls.HEAD_DIM,
            dtype=cls.DTYPE,
            pingpong=True,
        )

    @staticmethod
    def _teardown_state(state, sp_group):
        torch.cuda.synchronize()
        dist.barrier(sp_group)
        state.close()

    @staticmethod
    def _build_fused_tensors(q_weight, k_weight, v_weight, q_bias, k_bias, v_bias):
        with torch.no_grad():
            W_qkv_B = torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous().detach()
            if q_bias is not None:
                bias_B = torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous().detach()
            else:
                bias_B = None
        return W_qkv_B, bias_B

    @staticmethod
    def _run_path(
        *,
        x,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        norm_type,
        norm_q_weight,
        norm_k_weight,
        normalized_shape,
        eps,
        unpad_size,
        head_dim,
        sp_group,
        dispatch=None,
    ):
        return async_ulysses_qkv_projection(
            hidden_states=x,
            seq_dimension=1,
            head_dimension=2,
            q_weight=q_weight,
            q_bias=q_bias,
            k_weight=k_weight,
            k_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            norm_type=norm_type,
            norm_q_weight=norm_q_weight,
            norm_q_bias=None,
            norm_k_weight=norm_k_weight,
            norm_k_bias=None,
            normalized_shape=normalized_shape,
            eps=eps,
            unpadded_dim_size=unpad_size,
            head_dim=head_dim,
            group=sp_group,
            dispatch=dispatch,
        )

    @staticmethod
    def _backward_sum(q, k, v):
        # fp32 cast keeps the loss reduction out of bf16 noise; grads are unchanged.
        return q.float().sum() + k.float().sum() + v.float().sum()

    def _diagnose(self, label, ref, fused):
        if self.rank != 0 or ref is None or fused is None:
            return
        diff = (ref.detach().float() - fused.detach().float()).abs()
        denom = ref.detach().float().abs().clamp(min=1e-6)
        rel = diff / denom
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        mean_abs = diff.mean().item()
        over = (diff > self.ATOL).float().mean().item() * 100.0
        print(
            f"  {label:<24} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
            f"mean_abs={mean_abs:.3e}  ratio>atol={over:.2f}%",
            flush=True,
        )

    def _assert_close(self, label, ref, fused):
        if ref is None or fused is None:
            return
        torch.testing.assert_close(ref, fused, atol=self.ATOL, rtol=self.RTOL, msg=f"{label} mismatch")

    def _run_case(
        self,
        *,
        case_name: str,
        local_seq: int,
        nheads_q: int,
        nheads_kv: int,
        with_bias: bool,
        norm_type: str | None,
        unpad_size: int,
        seed: int,
    ):
        sp_group = self._get_process_group()
        self._maybe_skip_for_fused()
        self._maybe_skip_for_norm(norm_type)
        device = torch.device(get_device_id())
        torch.manual_seed(seed)

        x_local, q_w, k_w, v_w, q_b, k_b, v_b = self._build_inputs(
            sp_group,
            device,
            local_seq=local_seq,
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            with_bias=with_bias,
        )

        if norm_type == "rmsnorm":
            norm_q_weight = torch.randn(self.HEAD_DIM, dtype=self.DTYPE, device=device)
            norm_k_weight = torch.randn(self.HEAD_DIM, dtype=self.DTYPE, device=device)
            dist.broadcast(norm_q_weight, src=0)
            dist.broadcast(norm_k_weight, src=0)
            normalized_shape = self.HEAD_DIM
            eps = 1e-6
        else:
            norm_q_weight = norm_k_weight = None
            normalized_shape = None
            eps = None

        x_eager = x_local.clone().detach().requires_grad_(True)
        x_fused = x_local.clone().detach().requires_grad_(True)
        q_we, k_we, v_we = (self._clone_with_grad(t) for t in (q_w, k_w, v_w))
        q_wf, k_wf, v_wf = (self._clone_with_grad(t) for t in (q_w, k_w, v_w))
        q_be, k_be, v_be = (self._clone_with_grad(t) for t in (q_b, k_b, v_b))
        q_bf, k_bf, v_bf = (self._clone_with_grad(t) for t in (q_b, k_b, v_b))
        if norm_q_weight is not None:
            nq_we = norm_q_weight.clone().detach().requires_grad_(True)
            nk_we = norm_k_weight.clone().detach().requires_grad_(True)
            nq_wf = norm_q_weight.clone().detach().requires_grad_(True)
            nk_wf = norm_k_weight.clone().detach().requires_grad_(True)
        else:
            nq_we = nk_we = nq_wf = nk_wf = None

        ref_q, ref_k, ref_v = self._run_path(
            x=x_eager,
            q_weight=q_we,
            k_weight=k_we,
            v_weight=v_we,
            q_bias=q_be,
            k_bias=k_be,
            v_bias=v_be,
            norm_type=norm_type,
            norm_q_weight=nq_we,
            norm_k_weight=nk_we,
            normalized_shape=normalized_shape,
            eps=eps,
            unpad_size=unpad_size,
            head_dim=self.HEAD_DIM,
            sp_group=sp_group,
        )
        self._backward_sum(ref_q, ref_k, ref_v).backward()

        state = self._init_state(sp_group, device, local_seq=local_seq, nheads_q=nheads_q, nheads_kv=nheads_kv)
        try:
            W_qkv_B, bias_B = self._build_fused_tensors(q_wf, k_wf, v_wf, q_bf, k_bf, v_bf)
            fused_q, fused_k, fused_v = self._run_path(
                x=x_fused,
                q_weight=q_wf,
                k_weight=k_wf,
                v_weight=v_wf,
                q_bias=q_bf,
                k_bias=k_bf,
                v_bias=v_bf,
                norm_type=norm_type,
                norm_q_weight=nq_wf,
                norm_k_weight=nk_wf,
                normalized_shape=normalized_shape,
                eps=eps,
                unpad_size=unpad_size,
                head_dim=self.HEAD_DIM,
                sp_group=sp_group,
                dispatch=WsPushDispatch(state=state, W_qkv_B=W_qkv_B, bias_B=bias_B),
            )
            self._backward_sum(fused_q, fused_k, fused_v).backward()

            if self.rank == 0:
                print(
                    f"\n[fused-vs-eager:{case_name}] (local_seq={local_seq}, "
                    f"unpad_size={unpad_size}, nheads_q={nheads_q}, nheads_kv={nheads_kv}, "
                    f"norm={norm_type}, bias={with_bias})",
                    flush=True,
                )

            self._diagnose("q (forward)", ref_q, fused_q)
            self._diagnose("k (forward)", ref_k, fused_k)
            self._diagnose("v (forward)", ref_v, fused_v)
            self._assert_close("q (forward)", ref_q, fused_q)
            self._assert_close("k (forward)", ref_k, fused_k)
            self._assert_close("v (forward)", ref_v, fused_v)

            self._diagnose("q_weight.grad", q_we.grad, q_wf.grad)
            self._diagnose("k_weight.grad", k_we.grad, k_wf.grad)
            self._diagnose("v_weight.grad", v_we.grad, v_wf.grad)
            self._diagnose("x.grad", x_eager.grad, x_fused.grad)
            self._assert_close("q_weight.grad", q_we.grad, q_wf.grad)
            self._assert_close("k_weight.grad", k_we.grad, k_wf.grad)
            self._assert_close("v_weight.grad", v_we.grad, v_wf.grad)
            self._assert_close("x.grad", x_eager.grad, x_fused.grad)

            if with_bias:
                self._diagnose("q_bias.grad", q_be.grad, q_bf.grad)
                self._diagnose("k_bias.grad", k_be.grad, k_bf.grad)
                self._diagnose("v_bias.grad", v_be.grad, v_bf.grad)
                self._assert_close("q_bias.grad", q_be.grad, q_bf.grad)
                self._assert_close("k_bias.grad", k_be.grad, k_bf.grad)
                self._assert_close("v_bias.grad", v_be.grad, v_bf.grad)

            if norm_type == "rmsnorm":
                self._diagnose("norm_q_weight.grad", nq_we.grad, nq_wf.grad)
                self._diagnose("norm_k_weight.grad", nk_we.grad, nk_wf.grad)
                self._assert_close("norm_q_weight.grad", nq_we.grad, nq_wf.grad)
                self._assert_close("norm_k_weight.grad", nk_we.grad, nk_wf.grad)

            return (
                state,
                sp_group,
                fused_q.detach().clone(),
                {
                    "x_local": x_local,
                    "q_w": q_wf,
                    "k_w": k_wf,
                    "v_w": v_wf,
                    "W_qkv_B": W_qkv_B,
                    "unpad_size": unpad_size,
                },
            )
        except BaseException:
            self._teardown_state(state, sp_group)
            raise

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip fused-vs-eager numerical")
    def test_gqa_no_norm(self):
        ws = self.world_size
        nheads_q = max(8, ws * 2)
        nheads_kv = max(4, ws)
        state, sp_group, fused_q_snapshot, ctx = self._run_case(
            case_name="test_gqa_no_norm",
            local_seq=64,
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            with_bias=False,
            norm_type=None,
            unpad_size=64 * ws,
            seed=0xC0FFEE,
        )
        try:
            # Buffer-alias regression: if .clone() at async_ulysses.py:188-190 is dropped,
            # the second forward reuses peer_out_buf and corrupts fused_q_snapshot.
            with torch.no_grad():
                fused_q2, _, _ = self._run_path(
                    x=ctx["x_local"],
                    q_weight=ctx["q_w"].detach(),
                    k_weight=ctx["k_w"].detach(),
                    v_weight=ctx["v_w"].detach(),
                    q_bias=None,
                    k_bias=None,
                    v_bias=None,
                    norm_type=None,
                    norm_q_weight=None,
                    norm_k_weight=None,
                    normalized_shape=None,
                    eps=None,
                    unpad_size=ctx["unpad_size"],
                    head_dim=self.HEAD_DIM,
                    sp_group=sp_group,
                    dispatch=WsPushDispatch(state=state, W_qkv_B=ctx["W_qkv_B"], bias_B=None),
                )
                if self.rank == 0:
                    diff = (fused_q_snapshot.float() - fused_q2.float()).abs().max().item()
                    print(f"  buffer-alias check  max_abs={diff:.3e} (must be 0)", flush=True)
                torch.testing.assert_close(
                    fused_q_snapshot,
                    fused_q2,
                    atol=0.0,
                    rtol=0.0,
                    msg="fused_q snapshot diverged after second forward — peer_out_buf was reused",
                )
        finally:
            self._teardown_state(state, sp_group)

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip fused-vs-eager numerical")
    def test_gqa_with_rmsnorm_bias(self):
        ws = self.world_size
        nheads_q = max(8, ws * 2)
        nheads_kv = max(4, ws)
        state, sp_group, _, _ = self._run_case(
            case_name="test_gqa_with_rmsnorm_bias",
            local_seq=64,
            nheads_q=nheads_q,
            nheads_kv=nheads_kv,
            with_bias=True,
            norm_type="rmsnorm",
            unpad_size=64 * ws,
            seed=0xBADC0DE,
        )
        self._teardown_state(state, sp_group)

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip fused-vs-eager numerical")
    def test_seq_not_divisible_no_norm(self):
        ws = self.world_size
        local_seq = 64
        unpad_size = local_seq * ws - 1
        nheads = max(4, ws)
        state, sp_group, _, _ = self._run_case(
            case_name="test_seq_not_divisible_no_norm",
            local_seq=local_seq,
            nheads_q=nheads,
            nheads_kv=nheads,
            with_bias=False,
            norm_type=None,
            unpad_size=unpad_size,
            seed=0xDEADBEEF,
        )
        self._teardown_state(state, sp_group)

    @pytest.mark.skipif(get_torch_device().device_count() < 2, reason="device_count should be >= 2")
    @pytest.mark.skipif(is_torch_npu_available(), reason="npu skip fused-vs-eager numerical")
    def test_seq_not_divisible_with_rmsnorm(self):
        ws = self.world_size
        local_seq = 64
        unpad_size = local_seq * ws - 1
        nheads = max(4, ws)
        state, sp_group, _, _ = self._run_case(
            case_name="test_seq_not_divisible_with_rmsnorm",
            local_seq=local_seq,
            nheads_q=nheads,
            nheads_kv=nheads,
            with_bias=True,
            norm_type="rmsnorm",
            unpad_size=unpad_size,
            seed=0xFEEDFACE,
        )
        self._teardown_state(state, sp_group)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )
    run_tests()
