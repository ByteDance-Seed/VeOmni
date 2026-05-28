import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_id, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.async_ulysses import async_ulysses_qkv_projection
from veomni.distributed.sequence_parallel.data import slice_input_tensor
from veomni.ops.kernels.fused_ulysses_projection.ws_push import WsPushDispatch, init_ws_push_state
from veomni.utils.helper import set_seed
from veomni.utils.import_utils import is_torch_npu_available

from .utils import SequenceParallelTest


ATOL = 1e-2
RTOL = 1e-2


def _symm_mem_available() -> bool:
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        return True
    except ImportError:
        return False


class _FusedUlyssesPreAttnTestMixin:
    """Forward-numerics parity for ws_push fused vs eager AsyncUlyssesQKVProjection.

    Requires Hopper sm_90+ and torch.distributed._symmetric_memory. The fused kernel
    rejects KV-replication, so NHEADS_KV >= world_size."""

    BS = 2
    LOCAL_SEQ = 64
    NHEADS_Q = 8
    NHEADS_KV = 4
    HEAD_DIM = 64
    DTYPE = torch.bfloat16

    def _maybe_skip_for_fused(self):
        if is_torch_npu_available():
            self.skipTest("npu skip fused ulysses pre-attn")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if torch.cuda.device_count() < self.world_size:
            self.skipTest(f"device_count < world_size ({self.world_size})")
        if torch.cuda.get_device_capability(self.rank)[0] < 9:
            self.skipTest("fused kernel requires sm_90+ (Hopper)")
        if not _symm_mem_available():
            self.skipTest("torch.distributed._symmetric_memory not available")

    def _build_inputs(self, sp_group, device, *, with_bias: bool):
        hidden_dim = self.NHEADS_Q * self.HEAD_DIM
        seq_global = self.LOCAL_SEQ * dist.get_world_size(sp_group)

        full_x = torch.randn(self.BS, seq_global, hidden_dim, dtype=self.DTYPE, device=device)
        dist.broadcast(full_x, src=0)
        local_x = slice_input_tensor(full_x, dim=1, group=sp_group).contiguous()

        q_w = torch.randn(self.NHEADS_Q * self.HEAD_DIM, hidden_dim, dtype=self.DTYPE, device=device)
        k_w = torch.randn(self.NHEADS_KV * self.HEAD_DIM, hidden_dim, dtype=self.DTYPE, device=device)
        v_w = torch.randn(self.NHEADS_KV * self.HEAD_DIM, hidden_dim, dtype=self.DTYPE, device=device)
        for t in (q_w, k_w, v_w):
            dist.broadcast(t, src=0)

        if with_bias:
            q_b = torch.randn(self.NHEADS_Q * self.HEAD_DIM, dtype=self.DTYPE, device=device)
            k_b = torch.randn(self.NHEADS_KV * self.HEAD_DIM, dtype=self.DTYPE, device=device)
            v_b = torch.randn(self.NHEADS_KV * self.HEAD_DIM, dtype=self.DTYPE, device=device)
            for t in (q_b, k_b, v_b):
                dist.broadcast(t, src=0)
        else:
            q_b = k_b = v_b = None

        return local_x, q_w, k_w, v_w, q_b, k_b, v_b

    def _init_state(self, sp_group, device):
        # pingpong=True forces epi_tile_m=64 under quack 0.3.4 so each PackQKV epilogue
        # subtile maps to one batch; required when local_seq < 128.
        return init_ws_push_state(
            sp_group=sp_group,
            device=device,
            bs=self.BS,
            local_seq=self.LOCAL_SEQ,
            nheads_q=self.NHEADS_Q,
            nheads_k=self.NHEADS_KV,
            nheads_v=self.NHEADS_KV,
            head_dim=self.HEAD_DIM,
            dtype=self.DTYPE,
            pingpong=True,
        )

    @staticmethod
    def _teardown_state(state, sp_group):
        torch.cuda.synchronize()
        dist.barrier(sp_group)
        state.close()

    @staticmethod
    def _pack_fused_tensors(q_w, k_w, v_w, q_b, k_b, v_b):
        with torch.no_grad():
            W_qkv_B = torch.cat([q_w, k_w, v_w], dim=0).contiguous().detach()
            bias_B = torch.cat([q_b, k_b, v_b], dim=0).contiguous().detach() if q_b is not None else None
        return W_qkv_B, bias_B

    def _run_parity(self, *, with_bias: bool, with_norm: bool, seed: int):
        sp_group = self._get_process_group()
        self._maybe_skip_for_fused()
        if with_norm:
            self._maybe_skip_for_norm("rmsnorm")
        device = torch.device(get_device_id())
        torch.manual_seed(seed)

        x, q_w, k_w, v_w, q_b, k_b, v_b = self._build_inputs(sp_group, device, with_bias=with_bias)
        unpad_size = self.LOCAL_SEQ * self.world_size

        if with_norm:
            norm_q_weight = torch.randn(self.HEAD_DIM, dtype=self.DTYPE, device=device)
            norm_k_weight = torch.randn(self.HEAD_DIM, dtype=self.DTYPE, device=device)
            dist.broadcast(norm_q_weight, src=0)
            dist.broadcast(norm_k_weight, src=0)
            norm_kwargs = dict(
                norm_type="rmsnorm",
                norm_q_weight=norm_q_weight,
                norm_q_bias=None,
                norm_k_weight=norm_k_weight,
                norm_k_bias=None,
                normalized_shape=self.HEAD_DIM,
                eps=1e-6,
            )
        else:
            norm_kwargs = dict(
                norm_type=None,
                norm_q_weight=None,
                norm_q_bias=None,
                norm_k_weight=None,
                norm_k_bias=None,
                normalized_shape=None,
                eps=None,
            )

        common_kwargs = dict(
            hidden_states=x,
            seq_dimension=1,
            head_dimension=2,
            q_weight=q_w,
            q_bias=q_b,
            k_weight=k_w,
            k_bias=k_b,
            v_weight=v_w,
            v_bias=v_b,
            unpadded_dim_size=unpad_size,
            head_dim=self.HEAD_DIM,
            group=sp_group,
            **norm_kwargs,
        )

        ref_q, ref_k, ref_v = async_ulysses_qkv_projection(**common_kwargs, dispatch=None)

        state = self._init_state(sp_group, device)
        try:
            W_qkv_B, bias_B = self._pack_fused_tensors(q_w, k_w, v_w, q_b, k_b, v_b)
            fused_q, fused_k, fused_v = async_ulysses_qkv_projection(
                **common_kwargs,
                dispatch=WsPushDispatch(state=state, W_qkv_B=W_qkv_B, bias_B=bias_B),
            )

            torch.testing.assert_close(ref_q, fused_q, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(ref_k, fused_k, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(ref_v, fused_v, atol=ATOL, rtol=RTOL)
        finally:
            self._teardown_state(state, sp_group)

    def test_fused_pre_attn_no_bias_no_norm(self):
        self._run_parity(with_bias=False, with_norm=False, seed=0xC0FFEE)

    def test_fused_pre_attn_with_bias_and_rmsnorm(self):
        self._run_parity(with_bias=True, with_norm=True, seed=0xBADC0DE)


class FusedUlyssesPreAttn2GPUTest(_FusedUlyssesPreAttnTestMixin, SequenceParallelTest):
    @property
    def world_size(self) -> int:
        return 2


class FusedUlyssesPreAttn4GPUTest(_FusedUlyssesPreAttnTestMixin, SequenceParallelTest):
    @property
    def world_size(self) -> int:
        return 4


class FusedUlyssesPreAttn8GPUTest(_FusedUlyssesPreAttnTestMixin, SequenceParallelTest):
    NHEADS_Q = 16
    NHEADS_KV = 8

    @property
    def world_size(self) -> int:
        return 8


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    run_tests()
