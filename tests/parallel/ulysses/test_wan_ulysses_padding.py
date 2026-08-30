import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group
from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor_scale_grad
from veomni.distributed.sequence_parallel.utils import padding_tensor_for_seqeunce_parallel

from .utils import SequenceParallelTest


class WanUlyssesUnpatchifyPaddingTest(SequenceParallelTest):
    """Regression test for VeOmni issue #61.

    `WanModel.forward` slices its patchified sequence for Ulysses SP with
    `slice_input_tensor_scale_grad`, which floor-divides
    (`dim_size // seq_world_size`) instead of padding first. When the patchified
    token count isn't evenly divisible by the SP size -- e.g. a (f=21, h=60,
    w=45) grid gives f*h*w=56700 tokens, and 56700 % 8 == 4 under sp_size=8 --
    the remainder tokens are silently dropped: only 56696 of 56700 tokens ever
    reach any rank, and `unpatchify`'s einops rearrange then fails on the
    now-mismatched shape.

    This exercises the exact pad -> slice -> (identity op) -> gather -> unpad
    sequence `modeling_wan.py` now runs, and asserts the round trip is lossless
    and gradient-correct against a non-SP reference -- independent of the Wan
    model's weights, since the bug is in the SP primitives, not the model.
    """

    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_non_divisible_seq_len_round_trips_losslessly(self):
        self._get_process_group()
        sp_group = get_ulysses_sequence_parallel_group()

        # f*h*w for a (21, 60, 45) grid, sp_size=8, from issue #61 -- kept here at
        # world_size=4 (56700 % 4 == 0 would hide the bug), so instead use a
        # length with the same "not a clean multiple of sp_size" property at
        # world_size=4: 56700 - 56700 % 4 + 3 tokens short of the next multiple.
        seq_len = 56700 - (56700 % self.world_size) + (self.world_size - 1)
        assert seq_len % self.world_size != 0

        hidden_dim = 16
        full_input = torch.randn(1, seq_len, hidden_dim, device=get_device_type())
        c10d.broadcast(full_input, src=0)
        full_input.requires_grad_(True)

        # The exact sequence WanModel.forward now runs.
        unpadded_seq_len = full_input.shape[1]
        padded = padding_tensor_for_seqeunce_parallel(full_input, dim=1, group=sp_group)
        local = slice_input_tensor_scale_grad(padded, dim=1, group=sp_group)

        # A stand-in for the transformer blocks: any elementwise op so autograd
        # has something to differentiate through.
        local_out = local * 2.0

        gathered = gather_outputs(
            local_out,
            gather_dim=1,
            padding_dim=1,
            unpad_dim_size=unpadded_seq_len,
            scale_grad=False,
            group=sp_group,
        )

        assert gathered.shape[1] == seq_len, (
            f"expected the full {seq_len} tokens back, got {gathered.shape[1]} "
            f"(this is the exact failure mode of issue #61: floor-division in "
            f"slice_input_tensor_scale_grad silently drops the remainder)"
        )

        expected = full_input * 2.0
        torch.testing.assert_close(gathered, expected, atol=1e-6, rtol=1e-5)

        gathered.sum().backward()
        full_input_grad = full_input.grad.detach().clone()
        # every element's grad should be exactly 2.0 (d/dx of x*2, summed)
        torch.testing.assert_close(full_input_grad, torch.full_like(full_input_grad, 2.0))


if __name__ == "__main__":
    run_tests()
