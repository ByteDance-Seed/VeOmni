"""Multi-process regression test for Wan's sync-path Ulysses SP self-attention.

Issue #1140 (part 2): `SelfAttention.forward`'s sync path (`sp_async=False`, the only
path Wan actually uses) must redistribute Q/K/V via all-to-all before calling into
`AttentionModule`. Every rank then attends over the true full sequence with a subset
of heads, which is numerically equivalent to non-SP attention.

This drives `SelfAttention.forward` directly (bypassing `WanModel` and patchify/rope
grid setup) with `ulysses_size=world_size`. The default case uses a sequence length
that is a multiple of world_size; a second case pads a non-divisible length and
applies the global tail mask Wan builds in `WanModel.forward`.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


# A module-level `sys.exit(0)` here would raise SystemExit during collection and abort
# the whole pytest session (other test files included) on environments without the
# distributed backend, rather than just skipping this file -- use pytest's own skip
# mechanism instead.
if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    pytest.skip("c10d NCCL not available, skipping tests", allow_module_level=True)


from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state, get_parallel_state
from veomni.distributed.sequence_parallel.utils import padding_tensor_for_seqeunce_parallel
from veomni.models.transformers.wan.modeling_wan import SelfAttention, precompute_freqs_cis
from veomni.ops import resolve_op
from veomni.ops.config import resolve_op_impl

from .utils import SequenceParallelTest


def rope_apply_ref(x, freqs, head_dim):
    return resolve_op("rope", "wan", resolve_op_impl("rotary_pos_emb_implementation")).wrapper(
        x, freqs, head_dim=head_dim
    )


def _reference_out(self_attn, x_full, freqs_full, head_dim, attention_mask=None):
    q = self_attn.norm_q(self_attn.q(x_full))
    k = self_attn.norm_k(self_attn.k(x_full))
    v = self_attn.v(x_full)
    q = rope_apply_ref(q, freqs_full, head_dim)
    k = rope_apply_ref(k, freqs_full, head_dim)
    attn_out = self_attn.attn(
        q,
        k,
        v,
        last_loss=None,
        isSelfAttn=True,
        attention_mask=attention_mask,
        skip_ulysses=True,
    )
    return self_attn.o(attn_out)


class WanSelfAttentionUlyssesTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 4

    def _init_sp(self):
        _init_parallel_state(
            dp_size=1,
            ulysses_size=self.world_size,
            device_type=get_device_type(),
            name=None,
        )
        assert get_parallel_state().ulysses_enabled

    def _synced_self_attn(self, group, *, dim=64, num_heads=4, dtype=torch.float32):
        device = get_device_type()
        config = SimpleNamespace(_attn_implementation="eager")
        self_attn = SelfAttention(config, dim, num_heads).to(device=device, dtype=dtype)
        self._sync_model(self_attn.state_dict(), self.rank)
        for p in self_attn.parameters():
            c10d.broadcast(p.data, src=0, group=group)
        return self_attn

    def _broadcast_full_inputs(self, group, *, batch, full_seq_len, dim, head_dim, device, dtype):
        torch.manual_seed(0)
        x_full = torch.randn(batch, full_seq_len, dim, device=device, dtype=dtype)
        c10d.broadcast(x_full, src=0, group=group)
        freqs_full = precompute_freqs_cis(head_dim, end=full_seq_len).reshape(full_seq_len, 1, -1).to(device)
        return x_full, freqs_full

    def _gather_local(self, group, local_out, *, batch, unit, dim, device, dtype):
        chunks = [torch.empty(batch, unit, dim, device=device, dtype=dtype) for _ in range(self.world_size)]
        c10d.all_gather(chunks, local_out.contiguous(), group=group)
        return torch.cat(chunks, dim=1)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_sync_path_matches_non_sp_reference(self):
        group = self._get_process_group()
        try:
            self._init_sp()
            device = get_device_type()
            dtype = torch.float32
            dim, num_heads, full_seq_len, batch = 64, 4, 32, 2
            head_dim = dim // num_heads
            assert full_seq_len % self.world_size == 0

            self_attn = self._synced_self_attn(group, dim=dim, num_heads=num_heads, dtype=dtype)
            x_full, freqs_full = self._broadcast_full_inputs(
                group,
                batch=batch,
                full_seq_len=full_seq_len,
                dim=dim,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )

            with torch.no_grad():
                reference = _reference_out(self_attn, x_full, freqs_full, head_dim)

            unit = full_seq_len // self.world_size
            x_local = x_full[:, unit * self.rank : unit * (self.rank + 1), :].contiguous()
            freqs_local = freqs_full[unit * self.rank : unit * (self.rank + 1)].contiguous()
            with torch.no_grad():
                local_out = self_attn(x_local, freqs_local, cos=None, sin=None, last_loss=None, self_attn_mask=None)

            gathered = self._gather_local(
                group, local_out, batch=batch, unit=unit, dim=dim, device=device, dtype=dtype
            )
            # GPU float32 attention/matmul reduction order differs slightly between
            # "all heads, one process" (reference) and "subset of heads per rank,
            # concatenated" (actual) even though they're mathematically identical.
            torch.testing.assert_close(gathered, reference, atol=2e-4, rtol=2e-3)
        finally:
            clear_parallel_state()

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_sync_path_backward_matches_non_sp_reference(self):
        group = self._get_process_group()
        try:
            self._init_sp()
            device = get_device_type()
            dtype = torch.float32
            dim, num_heads, full_seq_len, batch = 64, 4, 32, 2
            head_dim = dim // num_heads

            self_attn = self._synced_self_attn(group, dim=dim, num_heads=num_heads, dtype=dtype)
            x_full, freqs_full = self._broadcast_full_inputs(
                group,
                batch=batch,
                full_seq_len=full_seq_len,
                dim=dim,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )

            reference = _reference_out(self_attn, x_full, freqs_full, head_dim)
            reference.sum().backward()
            ref_grads = {name: param.grad.detach().clone() for name, param in self_attn.named_parameters()}
            self_attn.zero_grad(set_to_none=True)

            unit = full_seq_len // self.world_size
            x_local = x_full[:, unit * self.rank : unit * (self.rank + 1), :].contiguous()
            freqs_local = freqs_full[unit * self.rank : unit * (self.rank + 1)].contiguous()
            local_out = self_attn(x_local, freqs_local, cos=None, sin=None, last_loss=None, self_attn_mask=None)
            local_out.sum().backward()
            for param in self_attn.parameters():
                c10d.all_reduce(param.grad, group=group)

            for name, param in self_attn.named_parameters():
                torch.testing.assert_close(param.grad, ref_grads[name], atol=2e-4, rtol=2e-3, msg=name)
        finally:
            clear_parallel_state()

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_sync_path_padded_seq_matches_non_sp_reference(self):
        group = self._get_process_group()
        try:
            self._init_sp()
            device = get_device_type()
            dtype = torch.float32
            dim, num_heads, full_seq_len, batch = 64, 4, 33, 2
            head_dim = dim // num_heads
            assert full_seq_len % self.world_size != 0

            self_attn = self._synced_self_attn(group, dim=dim, num_heads=num_heads, dtype=dtype)
            x_full, freqs_full = self._broadcast_full_inputs(
                group,
                batch=batch,
                full_seq_len=full_seq_len,
                dim=dim,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )

            with torch.no_grad():
                reference = _reference_out(self_attn, x_full, freqs_full, head_dim)

            x_padded = padding_tensor_for_seqeunce_parallel(x_full, dim=1, group=group)
            freqs_padded = padding_tensor_for_seqeunce_parallel(freqs_full, dim=0, group=group)
            padded_seq_len = x_padded.shape[1]
            pad_size = padded_seq_len - full_seq_len
            mask = torch.zeros(1, 1, 1, padded_seq_len, dtype=dtype, device=device)
            mask[..., padded_seq_len - pad_size :] = torch.finfo(dtype).min

            unit = padded_seq_len // self.world_size
            x_local = x_padded[:, unit * self.rank : unit * (self.rank + 1), :].contiguous()
            freqs_local = freqs_padded[unit * self.rank : unit * (self.rank + 1)].contiguous()
            with torch.no_grad():
                local_out = self_attn(x_local, freqs_local, cos=None, sin=None, last_loss=None, self_attn_mask=mask)

            gathered = self._gather_local(
                group, local_out, batch=batch, unit=unit, dim=dim, device=device, dtype=dtype
            )
            torch.testing.assert_close(gathered[:, :full_seq_len], reference, atol=2e-4, rtol=2e-3)
        finally:
            clear_parallel_state()


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
