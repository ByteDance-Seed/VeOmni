"""Multi-process regression test for Wan's sync-path Ulysses SP self-attention.

Issue #1140 (part 2): `SelfAttention.forward`'s sync path (`sp_async=False`, the only
path Wan actually uses) never redistributed Q/K/V via all-to-all before calling into
`AttentionModule`. Every rank's backend attention call therefore only ever saw its own
local SP shard as both query and key/value, instead of the true full sequence with a
subset of heads -- so Ulysses SP was not numerically equivalent to non-SP attention for
Wan on any of the three local backends (`eager`, `flash_attention_3`, `sageattention`).

This drives `SelfAttention.forward` directly (bypassing `WanModel` and patchify/rope
grid setup) with `ulysses_size=world_size` and no sequence padding (full_seq_len is a
multiple of world_size), and checks that the SP-sharded forward -- gathered back across
ranks -- exactly matches a single-process "full sequence, no SP" reference computed from
the same synced weights.
"""

import sys
from types import SimpleNamespace

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.parallel_state import clear_parallel_state, get_parallel_state, init_parallel_state
from veomni.models.transformers.wan.modeling_wan import SelfAttention, precompute_freqs_cis

from .utils import SequenceParallelTest


class WanSelfAttentionUlyssesTest(SequenceParallelTest):
    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_sync_path_matches_non_sp_reference(self):
        group = self._get_process_group()
        try:
            init_parallel_state(
                dp_size=1,
                ulysses_size=self.world_size,
                device_type=get_device_type(),
                name=None,
            )
            device = get_device_type()
            dtype = torch.float32
            dim, num_heads, full_seq_len, batch = 64, 4, 32, 2
            head_dim = dim // num_heads
            assert full_seq_len % self.world_size == 0  # no tail padding -- isolates the a2a fix

            config = SimpleNamespace(_attn_implementation="eager")
            self_attn = SelfAttention(config, dim, num_heads).to(device=device, dtype=dtype)
            self._sync_model(self_attn.state_dict(), self.rank)
            for p in self_attn.parameters():
                c10d.broadcast(p.data, src=0, group=group)

            torch.manual_seed(0)
            x_full = torch.randn(batch, full_seq_len, dim, device=device, dtype=dtype)
            c10d.broadcast(x_full, src=0, group=group)

            freqs_full = precompute_freqs_cis(head_dim, end=full_seq_len).reshape(full_seq_len, 1, -1).to(device)

            # Reference: exact sync-path math (norm -> rope -> eager attn -> out proj),
            # run directly on the FULL sequence with no SP redistribution at all.
            with torch.no_grad():
                q = self_attn.norm_q(self_attn.q(x_full))
                k = self_attn.norm_k(self_attn.k(x_full))
                v = self_attn.v(x_full)
                q = rope_apply_ref(q, freqs_full, head_dim)
                k = rope_apply_ref(k, freqs_full, head_dim)
                attn_out = self_attn.attn(q, k, v, last_loss=None, isSelfAttn=True, attention_mask=None)
                reference = self_attn.o(attn_out)

            # Actual: each rank runs the real SelfAttention.forward on its own local
            # shard, with Ulysses SP enabled -- exercising the all-to-all fix.
            unit = full_seq_len // self.world_size
            x_local = x_full[:, unit * self.rank : unit * (self.rank + 1), :].contiguous()
            freqs_local = freqs_full[unit * self.rank : unit * (self.rank + 1)].contiguous()

            assert get_parallel_state().ulysses_enabled
            with torch.no_grad():
                local_out = self_attn(x_local, freqs_local, cos=None, sin=None, last_loss=None, self_attn_mask=None)

            chunks = [torch.empty(batch, unit, dim, device=device, dtype=dtype) for _ in range(self.world_size)]
            c10d.all_gather(chunks, local_out.contiguous(), group=group)
            gathered = torch.cat(chunks, dim=1)

            # GPU float32 attention/matmul reduction order differs slightly between
            # "all heads, one process" (reference) and "subset of heads per rank,
            # concatenated" (actual) even though they're mathematically identical --
            # a couple ULPs of noise, not a correctness gap.
            torch.testing.assert_close(gathered, reference, atol=2e-4, rtol=2e-3)
        finally:
            clear_parallel_state()


def rope_apply_ref(x, freqs, head_dim):
    from veomni.models.transformers.wan.modeling_wan import rope_apply

    return rope_apply(x, freqs=freqs, cos=None, sin=None, head_dim=head_dim)


if __name__ == "__main__":
    run_tests()
